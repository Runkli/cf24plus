#!/usr/bin/env python3
import numpy as np
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Import SA optimizer from atlas_sa.py
from atlas_sa import ATLAS_SA_Optimizer

# Bayesian optimization imports
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logging.warning("scikit-optimize not available. Install with: pip install scikit-optimize")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("Plotting libraries not available. Install with: pip install matplotlib seaborn")

logger = logging.getLogger(__name__)

@dataclass
class TuningResults:
    """
    Results from SA parameter tuning with analysis and visualization methods
    """
    best_params: Dict[str, float]
    best_metrics: Dict[str, float]
    optimization_history: List[Dict]
    evaluation_count: int
    tuning_time: float
    convergence_trace: List[float]
    parameter_importance: Dict[str, float]
    
    def get_best_params(self) -> Dict[str, float]:
        """Get the best parameter configuration found"""
        return self.best_params.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of tuning performance"""
        return {
            'best_quality': self.best_metrics.get('quality_mean', 0),
            'quality_std': self.best_metrics.get('quality_std', 0),
            'best_time': self.best_metrics.get('time_mean', 0),
            'convergence_speed': self.best_metrics.get('convergence_speed', 0),
            'total_evaluations': self.evaluation_count,
            'tuning_time': self.tuning_time,
            'parameter_importance': self.parameter_importance
        }
    
    def plot_convergence(self):
        """Plot the convergence of the tuning process"""
        if not HAS_PLOTTING:
            logger.warning("Plotting not available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_trace)
        plt.title('Parameter Tuning Convergence')
        plt.xlabel('Evaluation Number')
        plt.ylabel('Best Objective Value')
        plt.grid(True)
        plt.show()
    
    def plot_parameter_importance(self):
        """Plot parameter importance rankings"""
        if not HAS_PLOTTING:
            logger.warning("Plotting not available")
            return
        
        params = list(self.parameter_importance.keys())
        importance = list(self.parameter_importance.values())
        
        plt.figure(figsize=(10, 6))
        plt.barh(params, importance)
        plt.title('Parameter Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def export_config(self, filename: str):
        """Export best parameters to a configuration file"""
        import json
        import numpy as np
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        config = {
            'best_parameters': convert_numpy_types(self.best_params),
            'performance_metrics': convert_numpy_types(self.best_metrics),
            'tuning_summary': convert_numpy_types(self.get_performance_summary())
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration exported to {filename}")


class SA_Parameter_Tuner:
    """
    Bayesian optimization-based parameter tuner for ATLAS SA optimizer
    
    Finds optimal SA parameters for a specific SNN clustering problem
    """
    
    def __init__(self, 
                 snn,
                 architecture,
                 initial_solution,
                 initial_blocks,
                 optimization_budget: int = 50,
                 validation_runs: int = 3,
                 random_state: int = 42):
        """
        Initialize the SA parameter tuner
        
        Args:
            snn: SNN graph to optimize
            architecture: Target architecture object
            initial_solution: Set of block IDs from initial ATLAS clustering
            initial_blocks: Dictionary mapping block IDs to Block objects
            optimization_budget: Total number of SA runs for parameter tuning
            validation_runs: Number of SA runs per parameter configuration for robustness
            random_state: Random seed for reproducibility
        """
        self.snn = snn
        self.architecture = architecture
        self.initial_solution = initial_solution
        self.initial_blocks = initial_blocks
        self.optimization_budget = optimization_budget
        self.validation_runs = validation_runs
        self.random_state = random_state
        
        # Tracking
        self.evaluation_count = 0
        self.optimization_history = []
        self.best_objective_trace = []
        self.current_best = float('-inf')
        
        # Default parameter bounds centered around known good parameters
        # Based on: initial_temp=50.0, alpha=0.95, iterations_per_temp=600, 
        # cluster_penalty_weight=15.0, neighbor_sample_size=50
        self.default_bounds = {
            'initial_temp': (40.0, 60.0),          # ±20% around 50.0
            'final_temp': (0.08, 0.12),            # ±20% around 0.1  
            'alpha': (0.93, 0.97),                 # ±2% around 0.95
            'iterations_per_temp': (480, 720),     # ±20% around 600
            'cluster_penalty_weight': (12.0, 18.0), # ±20% around 15.0
            'high_temp_threshold': (20.0, 40.0),   # Conservative range below initial_temp
            'neighbor_sample_size': (40, 60)       # ±20% around 50
        }
        
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize is required for parameter tuning. Install with: pip install scikit-optimize")
    
    def _get_parameter_space(self, param_bounds: Optional[Dict] = None):
        """
        Define the parameter space for Bayesian optimization
        
        Args:
            param_bounds: Custom parameter bounds, uses defaults for missing parameters
            
        Returns:
            List of skopt dimension objects
        """
        bounds = self.default_bounds.copy()
        if param_bounds:
            bounds.update(param_bounds)
        
        dimensions = [
            Real(bounds['initial_temp'][0], bounds['initial_temp'][1], name='initial_temp'),
            Real(bounds['final_temp'][0], bounds['final_temp'][1], name='final_temp'),
            Real(bounds['alpha'][0], bounds['alpha'][1], name='alpha'),
            Integer(bounds['iterations_per_temp'][0], bounds['iterations_per_temp'][1], name='iterations_per_temp'),
            Real(bounds['cluster_penalty_weight'][0], bounds['cluster_penalty_weight'][1], name='cluster_penalty_weight'),
            Real(bounds['high_temp_threshold'][0], bounds['high_temp_threshold'][1], name='high_temp_threshold'),
            Integer(bounds['neighbor_sample_size'][0], bounds['neighbor_sample_size'][1], name='neighbor_sample_size')
        ]
        
        return dimensions
    
    def _evaluate_parameter_set(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate a specific parameter configuration by running SA multiple times
        
        Args:
            params: Dictionary of SA parameters
            
        Returns:
            Dictionary of aggregated performance metrics
        """
        logger.debug(f"Evaluating parameter set: {params}")
        
        # Ensure parameters are properly typed for SA optimizer
        sa_params = self._format_parameters(params)
        
        results = []
        
        # Run SA multiple times for robustness assessment
        for run in range(self.validation_runs):
            try:
                # Create fresh copies for each run
                solution_copy = self.initial_solution.copy()
                blocks_copy = deepcopy(self.initial_blocks)
                
                # Create SA optimizer with current parameters
                sa_optimizer = ATLAS_SA_Optimizer(**sa_params)
                
                # Run SA optimization
                start_time = time.time()
                optimized_solution, optimized_blocks, sa_stats = sa_optimizer.optimize(
                    blocks_copy, solution_copy, self.architecture
                )
                run_time = time.time() - start_time
                
                # Verify we got a complete stats dictionary and add missing keys
                required_keys = ['final_quality', 'quality_improvement', 'cluster_reduction', 
                               'final_clusters', 'acceptance_rate']
                missing_keys = [key for key in required_keys if key not in sa_stats]
                
                if missing_keys:
                    logger.warning(f"SA stats missing keys: {missing_keys}")
                
                # Add 'made_improvement' key if missing (SA implementation doesn't include this)
                if 'made_improvement' not in sa_stats:
                    sa_stats['made_improvement'] = sa_stats.get('quality_improvement', 0) > 0
                
                # Store results for this run
                run_result = {
                    'final_quality': sa_stats.get('final_quality', -10.0),
                    'quality_improvement': sa_stats.get('quality_improvement', -5.0),
                    'optimization_time': run_time,
                    'cluster_reduction': sa_stats.get('cluster_reduction', 0),
                    'final_clusters': sa_stats.get('final_clusters', len(self.initial_solution)),
                    'acceptance_rate': sa_stats.get('acceptance_rate', 0.0),
                    'made_improvement': sa_stats['made_improvement']  # Now guaranteed to exist
                }
                
                results.append(run_result)
                self.evaluation_count += 1
                
            except Exception as e:
                logger.warning(f"SA run failed with parameters {params}: {e}")
                # Use penalty values for failed runs
                results.append({
                    'final_quality': -10.0,  # Large penalty
                    'quality_improvement': -5.0,
                    'optimization_time': 1000.0,  # Time penalty
                    'cluster_reduction': 0,
                    'final_clusters': len(self.initial_solution),
                    'acceptance_rate': 0.0,
                    'made_improvement': False
                })
                self.evaluation_count += 1
        
        # Aggregate results across runs
        aggregated = self._aggregate_results(results)
        
        # Store in optimization history
        self.optimization_history.append({
            'params': params.copy(),
            'metrics': aggregated.copy(),
            'individual_runs': results
        })
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """
        Aggregate results from multiple SA runs
        
        Args:
            results: List of result dictionaries from individual SA runs
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {'quality_mean': -10.0, 'quality_std': 0.0, 'time_mean': 1000.0}
        
        # Calculate statistics across runs
        quality_values = [r['final_quality'] for r in results]
        time_values = [r['optimization_time'] for r in results]
        improvement_values = [r['quality_improvement'] for r in results]
        cluster_reduction_values = [r['cluster_reduction'] for r in results]
        
        aggregated = {
            'quality_mean': np.mean(quality_values),
            'quality_std': np.std(quality_values),
            'quality_min': np.min(quality_values),
            'quality_max': np.max(quality_values),
            'time_mean': np.mean(time_values),
            'time_std': np.std(time_values),
            'improvement_mean': np.mean(improvement_values),
            'cluster_reduction_mean': np.mean(cluster_reduction_values),
            'success_rate': np.mean([r['made_improvement'] for r in results]),
            'consistency': 1.0 / (1.0 + np.std(quality_values))  # Higher for more consistent results
        }
        
        return aggregated
    
    def _format_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Format parameters to ensure correct types for SA optimizer
        
        Args:
            params: Raw parameters from Bayesian optimization
            
        Returns:
            Properly formatted parameters for SA optimizer
        """
        formatted = {}
        
        # Define expected types for each parameter
        float_params = ['initial_temp', 'final_temp', 'alpha', 'cluster_penalty_weight', 'high_temp_threshold']
        int_params = ['iterations_per_temp', 'neighbor_sample_size']
        
        for key, value in params.items():
            if key in float_params:
                formatted[key] = float(value)
            elif key in int_params:
                formatted[key] = int(round(value))  # Convert to int and round
            else:
                # Default to float for unknown parameters
                formatted[key] = float(value)
        
        return formatted
    
    def _calculate_objective(self, metrics: Dict[str, float], objective_type: str = 'quality') -> float:
        """
        Calculate the objective value for optimization
        
        Args:
            metrics: Aggregated metrics from SA runs
            objective_type: Type of objective ('quality', 'speed', 'balanced', 'robust')
            
        Returns:
            Objective value (higher is better)
        """
        if objective_type == 'quality':
            # Focus on solution quality
            return metrics['quality_mean']
        
        elif objective_type == 'speed':
            # Focus on fast convergence (lower time is better)
            return -metrics['time_mean'] / 100.0  # Normalize time
        
        elif objective_type == 'robust':
            # Focus on consistency across runs
            return metrics['quality_mean'] * metrics['consistency']
        
        elif objective_type == 'balanced':
            # Balance quality, speed, and robustness
            quality_score = metrics['quality_mean']
            speed_score = -metrics['time_mean'] / 100.0
            robustness_score = metrics['consistency']
            return 0.5 * quality_score + 0.3 * speed_score + 0.2 * robustness_score
        
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    def tune(self, 
             param_bounds: Optional[Dict] = None,
             objective_type: str = 'quality',
             acquisition_func: str = 'EI',
             n_initial_points: int = 10) -> TuningResults:
        """
        Run Bayesian optimization to find optimal SA parameters
        
        Args:
            param_bounds: Custom parameter bounds
            objective_type: Optimization objective ('quality', 'speed', 'balanced', 'robust')
            acquisition_func: Acquisition function ('EI', 'LCB', 'PI') - may not be supported in all skopt versions
            n_initial_points: Number of random initial evaluations
            
        Returns:
            TuningResults object with optimization results
        """
        logger.info(f"Starting SA parameter tuning with budget {self.optimization_budget}")
        logger.info(f"Objective: {objective_type}, Validation runs per config: {self.validation_runs}")
        
        start_time = time.time()
        
        # Reset tracking
        self.evaluation_count = 0
        self.optimization_history = []
        self.best_objective_trace = []
        self.current_best = float('-inf')
        
        # Define parameter space
        dimensions = self._get_parameter_space(param_bounds)
        
        # Create objective function for skopt
        @use_named_args(dimensions)
        def objective(**params):
            # Format parameters with correct types
            formatted_params = self._format_parameters(params)
            
            # Evaluate parameter set
            metrics = self._evaluate_parameter_set(formatted_params)
            
            # Calculate objective value
            obj_value = self._calculate_objective(metrics, objective_type)
            
            # Update best tracking
            if obj_value > self.current_best:
                self.current_best = obj_value
            
            self.best_objective_trace.append(self.current_best)
            
            # Progress logging
            logger.info(f"Evaluation {self.evaluation_count}/{self.optimization_budget}: "
                       f"objective={obj_value:.4f}, best={self.current_best:.4f}")
            
            # Return negative for minimization (skopt minimizes by default)
            return -obj_value
        
        # Run Bayesian optimization
        logger.info("Running Bayesian optimization...")
        
        try:
            # Try different optimization approaches based on availability
            n_calls = self.optimization_budget // self.validation_runs
            n_initial = min(n_initial_points, n_calls // 2)
            
            # First try gp_minimize with basic parameters (most compatible)
            try:
                result = gp_minimize(
                    func=objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    n_initial_points=n_initial,
                    random_state=self.random_state,
                    verbose=False
                )
                logger.info("Used Gaussian Process optimization")
                
            except Exception as gp_error:
                logger.warning(f"GP optimization failed: {gp_error}, trying forest-based optimization")
                
                # Fallback to forest_minimize (often more robust)
                result = forest_minimize(
                    func=objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    n_initial_points=n_initial,
                    random_state=self.random_state,
                    verbose=False
                )
                logger.info("Used Random Forest optimization")
            
            # Extract best parameters
            best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
            
            # Find best result in history
            best_entry = max(self.optimization_history, 
                           key=lambda x: self._calculate_objective(x['metrics'], objective_type))
            
            best_metrics = best_entry['metrics']
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            # Fallback to best found so far
            if self.optimization_history:
                best_entry = max(self.optimization_history, 
                               key=lambda x: self._calculate_objective(x['metrics'], objective_type))
                best_params = best_entry['params']
                best_metrics = best_entry['metrics']
            else:
                # Return default parameters as fallback
                best_params = self._get_fallback_parameters(dimensions)
                best_metrics = {'quality_mean': 0.0, 'time_mean': 0.0}
                logger.warning("Using fallback parameters due to optimization failure")
        
        tuning_time = time.time() - start_time
        
        # Calculate parameter importance (simplified)
        parameter_importance = self._calculate_parameter_importance()
        
        # Create results object
        results = TuningResults(
            best_params=best_params,
            best_metrics=best_metrics,
            optimization_history=self.optimization_history,
            evaluation_count=self.evaluation_count,
            tuning_time=tuning_time,
            convergence_trace=self.best_objective_trace,
            parameter_importance=parameter_importance
        )
        
        logger.info(f"Parameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best performance: {best_metrics}")
        
        return results
    
    def _get_fallback_parameters(self, dimensions) -> Dict[str, float]:
        """
        Get reasonable fallback parameters when optimization fails
        
        Args:
            dimensions: Parameter space dimensions
            
        Returns:
            Dictionary of fallback parameter values
        """
        fallback = {}
        
        for dim in dimensions:
            if hasattr(dim, 'low') and hasattr(dim, 'high'):
                # For Real dimensions, use middle value
                fallback[dim.name] = (dim.low + dim.high) / 2
            elif hasattr(dim, 'bounds'):
                # For Integer dimensions, use middle value
                low, high = dim.bounds
                fallback[dim.name] = int((low + high) / 2)
            else:
                # Default values based on known good parameters
                defaults = {
                    'initial_temp': 50.0,          # Known good value
                    'final_temp': 0.1,             # Known good value
                    'alpha': 0.95,                 # Known good value
                    'iterations_per_temp': 600,    # Known good value
                    'cluster_penalty_weight': 15.0, # Known good value
                    'high_temp_threshold': 25.0,   # Conservative default
                    'neighbor_sample_size': 50     # Known good value
                }
                fallback[dim.name] = defaults.get(dim.name, 1.0)
        
        return fallback
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance based on optimization history
        Simple correlation-based approach
        """
        if len(self.optimization_history) < 10:
            return {}
        
        import pandas as pd
        
        # Extract parameter values and objective values
        param_data = []
        objective_values = []
        
        for entry in self.optimization_history:
            param_data.append(entry['params'])
            objective_values.append(self._calculate_objective(entry['metrics'], 'quality'))
        
        df_params = pd.DataFrame(param_data)
        df_params['objective'] = objective_values
        
        # Calculate correlations
        correlations = df_params.corr()['objective'].abs().drop('objective')
        
        return correlations.to_dict()
    
    def quick_tune(self, objective_type: str = 'quality', budget_fraction: float = 0.6) -> Dict[str, float]:
        """
        Quick parameter tuning with reduced budget
        
        Args:
            objective_type: Optimization objective
            budget_fraction: Fraction of full budget to use
            
        Returns:
            Dictionary of best parameters found
        """
        # Temporarily reduce budget
        original_budget = self.optimization_budget
        self.optimization_budget = int(original_budget * budget_fraction)
        
        try:
            results = self.tune(objective_type=objective_type, n_initial_points=5)
            return results.get_best_params()
        finally:
            # Restore original budget
            self.optimization_budget = original_budget
    
    def cross_validate(self, params: Dict[str, float], num_runs: int = 10) -> Dict[str, float]:
        """
        Cross-validate specific parameters with multiple runs
        
        Args:
            params: SA parameters to validate
            num_runs: Number of validation runs
            
        Returns:
            Detailed performance statistics
        """
        logger.info(f"Cross-validating parameters with {num_runs} runs")
        
        original_validation_runs = self.validation_runs
        self.validation_runs = num_runs
        
        try:
            metrics = self._evaluate_parameter_set(params)
            return metrics
        finally:
            self.validation_runs = original_validation_runs


def main():
    """Example usage of SA parameter tuner"""
    import sys
    sys.path.append('.')
    
    try:
        # Import required components
        from atlas_base import ATLAS, OptimizedCrossbar
        
        # Load or create test problem
        try:
            from reader import AdarshaReader
            
            model_name = "LeNet"
            model_path = f'../disco_libs/SNN_db/{model_name}'
            conn_file = model_path + '.connection_info.txt' 
            spk_file = model_path + '.spike_info.txt' 
            
            reader = AdarshaReader()
            model, aer = reader.read(conn_file, spk_file)
            reader.addTraffic(model, aer)
            
            snn = model
            logger.info(f"Loaded {model_name} with {len(snn.nodes())} neurons")
            
        except ImportError:
            # Create synthetic network
            import networkx as nx
            import random
            
            logger.info("Creating synthetic SNN for testing")
            snn = nx.DiGraph()
            
            # Create layered network
            neurons_per_layer = [300, 150, 75]
            connection_prob = 0.1
            
            neuron_id = 0
            for layer, layer_size in enumerate(neurons_per_layer):
                for i in range(layer_size):
                    snn.add_node(neuron_id, layer=layer)
                    neuron_id += 1
            
            # Add connections
            for layer in range(len(neurons_per_layer) - 1):
                source_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer]
                target_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer + 1]
                
                for source in source_neurons:
                    for target in target_neurons:
                        if random.random() < connection_prob:
                            snn.add_edge(source, target, traffic=random.randint(1, 10))
            
            logger.info(f"Created synthetic SNN with {len(snn.nodes())} neurons and {len(snn.edges())} edges")
        
        # Create architecture
        crossbar_size = 1024
        arch = OptimizedCrossbar(crossbar_size)
        
        # Run initial ATLAS clustering
        logger.info("Running initial ATLAS clustering...")
        atlas = ATLAS(snn, arch)
        
        # Get initial solution
        atlas._create_minimum_blocks()
        atlas._build_atlas_graph()
        atlas._run_greedy_phase(target=1000)  # Target 100 clusters
        
        initial_solution = atlas.get_current_solution()
        initial_blocks = atlas.blocks.copy()
        
        logger.info(f"Initial ATLAS solution: {len(initial_solution)} clusters")
        
        # Create parameter tuner
        tuner = SA_Parameter_Tuner(
            snn=snn,
            architecture=arch,
            initial_solution=initial_solution,
            initial_blocks=initial_blocks,
            optimization_budget=30,  # Small budget for demo
            validation_runs=2
        )
        
        # Run parameter tuning
        logger.info("Starting parameter tuning...")
        results = tuner.tune(objective_type='quality')
        
        # Print results
        print("\n" + "="*50)
        print("PARAMETER TUNING RESULTS")
        print("="*50)
        
        print(f"\nBest Parameters:")
        for param, value in results.best_params.items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\nPerformance Metrics:")
        for metric, value in results.best_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nTuning Summary:")
        summary = results.get_performance_summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Test the tuned parameters
        logger.info("Testing tuned parameters...")
        from atlas_sa import ATLAS_SA_Optimizer
        
        # Format parameters properly before using
        formatted_best_params = tuner._format_parameters(results.best_params)
        sa_optimizer = ATLAS_SA_Optimizer(**formatted_best_params)
        optimized_solution, optimized_blocks, sa_stats = sa_optimizer.optimize(
            initial_blocks.copy(), initial_solution.copy(), arch
        )
        
        print(f"\nFinal Test Results:")
        print(f"  Clusters reduced from {len(initial_solution)} to {len(optimized_solution)}")
        print(f"  Final quality: {sa_stats['final_quality']:.4f}")
        print(f"  Quality improvement: {sa_stats['quality_improvement']:.4f}")
        
        # Export configuration
        results.export_config('best_sa_config.json')
        
        # Show plots if available
        if HAS_PLOTTING:
            results.plot_convergence()
            results.plot_parameter_importance()
        
    except ImportError as e:
        logger.error(f"Required imports not available: {e}")
        logger.info("Make sure atlas_base.py and atlas_sa.py are available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
