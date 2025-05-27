#!/usr/bin/env python3
import os
import pickle
import logging
import time
from copy import deepcopy

# Import required components
from atlas_base import ATLAS, OptimizedCrossbar
from atlas_sa import ATLAS_SA_Optimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreedyTargetOptimizer:
    """
    Optimizes the greedy target parameter to minimize final clusters after SA.
    Uses binary search with caching to efficiently explore the target space.
    """
    
    def __init__(self, snn, architecture, cache_dir="./atlas_cache", crossbar_size=1024):
        """
        Initialize the greedy target optimizer
        
        Args:
            snn: SNN graph to optimize
            architecture: Target architecture object
            cache_dir: Directory to store cached results
            crossbar_size: Size of crossbar architecture
        """
        self.snn = snn
        self.architecture = architecture
        self.cache_dir = cache_dir
        self.crossbar_size = crossbar_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Fixed SA parameters (known to work well)
        self.sa_params = {
            'initial_temp': 50.0,
            'final_temp': 0.1,
            'alpha': 0.95,
            'iterations_per_temp': 600,
            'cluster_penalty_weight': 15.0,
            'neighbor_sample_size': 50
        }
        
        # Results tracking
        self.results = {}  # target -> final_clusters_after_sa
        self.best_target = None
        self.best_final_clusters = float('inf')
        self.best_solution = None
        self.best_blocks = None
        
        logger.info(f"Initialized Greedy Target Optimizer with crossbar size {crossbar_size}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"SA parameters: {self.sa_params}")
    
    def optimize(self, step_size=None):
        """
        Main optimization method using linear sweep from B//4 down to A
        
        Args:
            step_size: Step size for sweep (defaults to A if None)
            
        Returns:
            dict: Results including best target and final solution
        """
        logger.info("Starting greedy target optimization with linear sweep")
        start_time = time.time()
        
        # Step 1: Run full greedy to get bounds
        logger.info("Step 1: Running full greedy to establish bounds")
        A, B = self._run_full_greedy()
        
        logger.info(f"Full greedy results: A (final clusters) = {A}, B (initial clusters) = {B}")
        
        # Step 2: Set up linear sweep parameters
        upper_bound = B // 4  # Start from B//4
        step = step_size if step_size is not None else A  # Step size = A
        
        logger.info(f"Linear sweep from {upper_bound} down to {A} in steps of {step}")
        
        if upper_bound <= A:
            logger.warning("Upper bound <= A, using full greedy result")
            return self._format_results(time.time() - start_time)
        
        # Step 3: Linear sweep from upper_bound down to A
        current_target = upper_bound
        step_num = 1
        
        while current_target >= A:
            logger.info(f"\nStep {step_num}: Testing target {current_target}")
            
            # Test this target
            final_clusters = self._test_target(current_target)
            
            logger.info(f"Step {step_num} - Target {current_target} → {final_clusters} final clusters")
            
            # Update best if this is better
            if final_clusters < self.best_final_clusters:
                self.best_final_clusters = final_clusters
                self.best_target = current_target
                logger.info(f"NEW BEST: Step {step_num} - Target {current_target} with {final_clusters} final clusters")
            
            # Move to next target
            current_target -= step
            step_num += 1
        
        # Also test the final bound A
        if A not in self.results:
            logger.info(f"\nFinal step: Testing target {A}")
            final_clusters = self._test_target(A)
            logger.info(f"Final step - Target {A} → {final_clusters} final clusters")
            
            if final_clusters < self.best_final_clusters:
                self.best_final_clusters = final_clusters
                self.best_target = A
                logger.info(f"NEW BEST: Final step - Target {A} with {final_clusters} final clusters")
        
        optimization_time = time.time() - start_time
        
        logger.info(f"\nLinear sweep completed after {step_num-1} steps")
        logger.info(f"Best target found: {self.best_target} with {self.best_final_clusters} final clusters")
        
        return self._format_results(optimization_time)
    
    def _run_full_greedy(self):
        """
        Run greedy algorithm to completion to establish bounds
        
        Returns:
            tuple: (final_clusters, initial_clusters)
        """
        # Create fresh ATLAS instance
        atlas = ATLAS(self.snn, self.architecture)
        
        # Initialize
        atlas._create_minimum_blocks()
        atlas._build_atlas_graph()
        
        initial_clusters = len(atlas.get_current_solution())
        logger.info(f"Initial minimum blocks: {initial_clusters}")
        
        # Run greedy to completion
        merges = atlas._run_greedy_phase()  # No target = run to completion
        
        final_clusters = len(atlas.get_current_solution())
        logger.info(f"Full greedy completed: {merges} merges, {final_clusters} final clusters")
        
        # Cache the full greedy result
        self._save_atlas_state(atlas, final_clusters, "full_greedy")
        
        return final_clusters, initial_clusters
    
    def _test_target(self, target):
        """
        Test a specific greedy target and return final clusters after SA
        
        Args:
            target: Target number of clusters for greedy phase
            
        Returns:
            int: Final number of clusters after SA
        """
        # Check if we already tested this target
        if target in self.results:
            logger.info(f"Using cached result for target {target}")
            return self.results[target]
        
        # Get atlas state at this target
        atlas, solution, blocks = self._get_atlas_at_target(target)
        
        if atlas is None:
            logger.warning(f"Could not reach target {target}, skipping")
            return float('inf')
        
        logger.info(f"Running SA on {len(solution)} clusters from target {target}")
        
        # Run SA optimization
        sa_optimizer = ATLAS_SA_Optimizer(**self.sa_params)
        
        try:
            optimized_solution, optimized_blocks, sa_stats = sa_optimizer.optimize(
                blocks, solution, self.architecture
            )
            
            final_clusters = len(optimized_solution)
            
            logger.info(f"SA completed: {len(solution)} → {final_clusters} clusters")
            logger.info(f"SA stats: quality_improvement={sa_stats.get('quality_improvement', 0):.4f}, "
                       f"acceptance_rate={sa_stats.get('acceptance_rate', 0):.3f}")
            
            # Store results
            self.results[target] = final_clusters
            
            # Update best if this is better
            if final_clusters < self.best_final_clusters:
                self.best_final_clusters = final_clusters
                self.best_target = target
                self.best_solution = optimized_solution
                self.best_blocks = optimized_blocks
            
            return final_clusters
            
        except Exception as e:
            logger.error(f"SA failed for target {target}: {e}")
            self.results[target] = float('inf')
            return float('inf')
    
    def _get_atlas_at_target(self, target):
        """
        Get ATLAS state at specified target, using caching when possible
        
        Args:
            target: Target number of clusters
            
        Returns:
            tuple: (atlas, solution, blocks) or (None, None, None) if failed
        """
        # Find the best cached state to start from
        cached_targets = self._get_cached_targets()
        
        if not cached_targets:
            # No cache, start from beginning
            return self._create_atlas_at_target_from_scratch(target)
        
        # Find the LARGEST cached target (since greedy reduces clusters over time)
        # We can only continue greedy from a larger number to a smaller number
        suitable_targets = [t for t in cached_targets if t >= target]
        
        if suitable_targets:
            # Start from the LARGEST suitable cached target (closest to where we are)
            start_target = max(suitable_targets)
            logger.info(f"Starting from cached target {start_target} to reach {target}")
            
            # Load cached state
            atlas, solution, blocks = self._load_atlas_state(start_target)
            
            if atlas is None:
                logger.error(f"Failed to load cached state for {start_target}")
                return self._create_atlas_at_target_from_scratch(target)
            
            if start_target == target:
                # Exact match
                return atlas, solution, blocks
            else:
                # Continue greedy from cached state to target
                current_clusters = len(atlas.get_current_solution())
                logger.info(f"Continuing greedy from {current_clusters} clusters to target {target}")
                
                if current_clusters <= target:
                    logger.warning(f"Cached state has {current_clusters} clusters, already at or below target {target}")
                    return atlas, atlas.get_current_solution(), atlas.blocks
                
                # Run greedy to target
                atlas._run_greedy_phase(target=target)
                solution = atlas.get_current_solution()
                blocks = atlas.blocks
                
                logger.info(f"Greedy continued to {len(solution)} clusters")
                
                # Cache this new state
                self._save_atlas_state(atlas, target)
                
                return atlas, solution, blocks
        else:
            # All cached targets are smaller than our target
            # We need to start from scratch since we can't go backwards in greedy
            logger.info(f"No suitable cached targets for {target} (all cached targets are smaller), starting from scratch")
            return self._create_atlas_at_target_from_scratch(target)
    
    def _create_atlas_at_target_from_scratch(self, target):
        """
        Create ATLAS state at target from scratch
        
        Args:
            target: Target number of clusters
            
        Returns:
            tuple: (atlas, solution, blocks)
        """
        logger.info(f"Creating ATLAS state for target {target} from scratch")
        
        # Create fresh ATLAS instance
        atlas = ATLAS(self.snn, self.architecture)
        
        # Initialize
        atlas._create_minimum_blocks()
        atlas._build_atlas_graph()
        
        # Run greedy to target
        atlas._run_greedy_phase(target=target)
        
        solution = atlas.get_current_solution()
        blocks = atlas.blocks
        
        # Cache this state
        self._save_atlas_state(atlas, target)
        
        return atlas, solution, blocks
    
    def _save_atlas_state(self, atlas, target, suffix=""):
        """
        Save ATLAS state to cache
        
        Args:
            atlas: ATLAS instance
            target: Target number of clusters (for filename)
            suffix: Optional suffix for filename
        """
        filename = f"atlas_cache_target_{target}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pkl"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        cache_data = {
            'atlas_graph': deepcopy(atlas.atlas_graph),
            'blocks': deepcopy(atlas.blocks),
            'solution': atlas.get_current_solution(),
            'target': target,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Saved ATLAS state for target {target} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save cache for target {target}: {e}")
    
    def _load_atlas_state(self, target):
        """
        Load ATLAS state from cache
        
        Args:
            target: Target number of clusters
            
        Returns:
            tuple: (atlas, solution, blocks)
        """
        filename = f"atlas_cache_target_{target}.pkl"
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Reconstruct ATLAS instance
            atlas = ATLAS(self.snn, self.architecture)
            atlas.atlas_graph = cache_data['atlas_graph']
            atlas.blocks = cache_data['blocks']
            
            solution = cache_data['solution']
            blocks = cache_data['blocks']
            
            logger.debug(f"Loaded ATLAS state for target {target} from {filepath}")
            return atlas, solution, blocks
            
        except Exception as e:
            logger.error(f"Failed to load cache for target {target}: {e}")
            return None, None, None
    
    def _get_cached_targets(self):
        """
        Get list of cached targets
        
        Returns:
            list: List of cached target values
        """
        cached_targets = []
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("atlas_cache_target_") and filename.endswith(".pkl"):
                    # Extract target number from filename
                    parts = filename.replace("atlas_cache_target_", "").replace(".pkl", "")
                    # Handle suffixes like "_full_greedy"
                    target_str = parts.split("_")[0]
                    try:
                        target = int(target_str)
                        cached_targets.append(target)
                    except ValueError:
                        continue
                        
        except FileNotFoundError:
            pass
        
        return sorted(cached_targets)
    
    def _format_results(self, optimization_time):
        """
        Format final results
        
        Args:
            optimization_time: Total optimization time
            
        Returns:
            dict: Formatted results
        """
        return {
            'best_target': self.best_target,
            'best_final_clusters': self.best_final_clusters,
            'best_solution': self.best_solution,
            'best_blocks': self.best_blocks,
            'all_results': self.results.copy(),
            'optimization_time': optimization_time,
            'sa_parameters': self.sa_params.copy(),
            'num_targets_tested': len(self.results)
        }
    
    def clear_cache(self):
        """Clear all cached results"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("atlas_cache_target_"):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def print_results(self, results):
        """Print comprehensive results"""
        print("\n" + "="*70)
        print("GREEDY TARGET OPTIMIZATION RESULTS")
        print("="*70)
        
        print(f"Best Target: {results['best_target']} clusters")
        print(f"Best Final Result: {results['best_final_clusters']} clusters")
        print(f"Optimization Time: {results['optimization_time']:.2f} seconds")
        print(f"Targets Tested: {results['num_targets_tested']}")
        
        print(f"\nSA Parameters Used:")
        for param, value in results['sa_parameters'].items():
            print(f"  {param}: {value}")
        
        print(f"\nAll Results (Target → Final Clusters):")
        for target in sorted(results['all_results'].keys()):
            final = results['all_results'][target]
            marker = " ←BEST" if target == results['best_target'] else ""
            print(f"  {target:4d} → {final:4d}{marker}")
        
        print("="*70)


def main():
    """Main execution"""
    try:
        # Import reader
        from reader import AdarshaReader
        
        # Load SNN model
        model_name = "LeNet"
        model_path = f'../disco_libs/SNN_db/{model_name}'
        conn_file = model_path + '.connection_info.txt' 
        spk_file = model_path + '.spike_info.txt' 
        
        reader = AdarshaReader()
        model, aer = reader.read(conn_file, spk_file)
        reader.addTraffic(model, aer)
        
        snn = model
        logger.info(f"Loaded {model_name} with {len(snn.nodes())} neurons and {len(snn.edges())} edges")
        
    except ImportError:
        # Create synthetic SNN for testing
        import networkx as nx
        import random
        
        logger.info("Creating synthetic SNN for testing")
        snn = nx.DiGraph()
        
        # Create layered network
        neurons_per_layer = [1000, 500, 250]
        connection_prob = 0.05
        
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
    
    # Create optimizer
    optimizer = GreedyTargetOptimizer(snn, arch, crossbar_size=crossbar_size)
    
    # Run optimization
    logger.info("Starting greedy target optimization...")
    results = optimizer.optimize()  # Uses default step_size=A
    
    # Print results
    optimizer.print_results(results)
    
    # Save results to file
    import json
    with open("greedy_target_optimization_results.json", "w") as f:
        # Convert sets to lists for JSON serialization
        json_results = results.copy()
        if json_results['best_solution']:
            json_results['best_solution'] = list(json_results['best_solution'])
        json_results['best_blocks'] = "omitted_for_size"  # Too large for JSON
        
        json.dump(json_results, f, indent=2)
    
    logger.info("Results saved to greedy_target_optimization_results.json")


if __name__ == "__main__":
    main()
