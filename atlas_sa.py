#!/usr/bin/env python3
import math
import random
import time
import logging
import pyroaring
import pickle
from copy import deepcopy

logger = logging.getLogger(__name__)

class ATLAS_SA_Optimizer:
    """
    Simulated Annealing optimizer for ATLAS clustering results.
    Works as a separate optimization pass to refine greedy ATLAS solutions.
    """
    
    def __init__(self, 
                 initial_temp=100.0,
                 final_temp=0.1,
                 alpha=0.95,
                 iterations_per_temp=500,
                 cluster_penalty_weight=20.0,
                 neighbor_sample_size=100,
                 high_temp_threshold=50.0):
        """
        Initialize SA optimizer with parameters
        
        Args:
            initial_temp: Starting temperature for SA
            final_temp: Final temperature (stopping criterion)
            alpha: Temperature cooling rate (multiplicative)
            iterations_per_temp: Number of iterations at each temperature
            cluster_penalty_weight: Weight for cluster count penalty in quality function
            neighbor_sample_size: Number of blocks to sample when finding neighbors
            high_temp_threshold: Temperature threshold for exploration vs exploitation
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.cluster_penalty_weight = cluster_penalty_weight
        self.neighbor_sample_size = neighbor_sample_size
        self.high_temp_threshold = high_temp_threshold
        
        # Block ID counter - will be initialized in optimize() method
        self.next_block_id = 0
        
        # Statistics tracking
        self.stats = {
            'optimization_time': 0,
            'total_iterations': 0,
            'total_acceptances': 0,
            'total_rejections': 0,
            'temperature_levels': 0,
            'initial_quality': 0,
            'final_quality': 0,
            'initial_clusters': 0,
            'final_clusters': 0,
            'quality_improvement': 0,
            'cluster_reduction': 0,
            'acceptance_rate': 0,
            'best_quality_found': float('-inf'),
            'best_solution_found': None
        }
        
    def optimize(self, blocks, initial_solution, architecture, starting_temp=None):
        """
        Main optimization method
        
        Args:
            blocks: Dictionary mapping block_id -> Block object
            initial_solution: Set of block IDs representing current solution
            architecture: Architecture object with valid() and util() methods
            starting_temp: Optional override for starting temperature
            
        Returns:
            tuple: (optimized_solution, optimized_blocks, stats)
        """
        start_time = time.time()
        
        if starting_temp is not None:
            self.initial_temp = starting_temp
        
        # Initialize next_block_id to start from 0 and increment sequentially
        # Find the maximum existing block ID to avoid conflicts
        if blocks:
            max_existing_id = max(blocks.keys())
            self.next_block_id = max_existing_id + 1
        else:
            self.next_block_id = 0
        
        # Initialize solution and blocks
        current_solution = set(initial_solution)
        current_blocks = deepcopy(blocks)
        
        # Calculate initial quality
        initial_quality = self._calculate_solution_quality(current_solution, current_blocks, architecture)
        
        # Track best solution found
        best_solution = set(current_solution)
        best_blocks = deepcopy(current_blocks)
        best_quality = initial_quality
        
        # Initialize statistics
        self.stats['initial_quality'] = initial_quality
        self.stats['initial_clusters'] = len(current_solution)
        self.stats['best_quality_found'] = best_quality
        self.stats['best_solution_found'] = set(best_solution)
        
        logger.info(f"SA Optimization starting - Initial: {len(current_solution)} clusters, quality: {initial_quality:.4f}")
        
        # SA main loop
        current_temp = self.initial_temp
        total_iterations = 0
        total_acceptances = 0
        temperature_attempts = 0
        
        while current_temp > self.final_temp:
            self.stats['temperature_levels'] += 1
            temp_acceptances = 0
            
            logger.info(f"SA Temperature: {current_temp:.2f}, Clusters: {len(current_solution)}")
            
            for iteration in range(self.iterations_per_temp):
                total_iterations += 1
                
                # Generate neighbor solution
                neighbor_solution, neighbor_blocks = self._get_neighbor_solution(
                    current_solution, current_blocks, architecture, current_temp
                )
                
                if neighbor_solution is None:
                    continue
                    
                # Calculate quality of neighbor
                neighbor_quality = self._calculate_solution_quality(
                    neighbor_solution, neighbor_blocks, architecture
                )
                
                # Calculate current quality
                current_quality = self._calculate_solution_quality(
                    current_solution, current_blocks, architecture
                )
                
                # Accept or reject move
                if self._accept_move(current_quality, neighbor_quality, current_temp):
                    current_solution = neighbor_solution
                    current_blocks = neighbor_blocks
                    total_acceptances += 1
                    temp_acceptances += 1
                    
                    # Update best solution if necessary
                    if neighbor_quality > best_quality:
                        best_quality = neighbor_quality
                        best_solution = set(neighbor_solution)
                        best_blocks = deepcopy(neighbor_blocks)
                        self.stats['best_quality_found'] = best_quality
                        self.stats['best_solution_found'] = set(best_solution)
                        
                        logger.info(f"New best solution: {len(best_solution)} clusters, quality: {best_quality:.4f}")
            
            logger.info(f"Temperature {current_temp:.2f}: {temp_acceptances}/{self.iterations_per_temp} accepted")
            
            # Cool down
            current_temp *= self.alpha
            if temp_acceptances > 0:
                temperature_attempts = 0

            if temp_acceptances == 0:
                temperature_attempts += 1

            if temperature_attempts >= 10:
                logger.info(f"Stopping search after {temperature_attempts} attempts with no improvement")
                break

        
        # Finalize statistics
        optimization_time = time.time() - start_time
        final_quality = self._calculate_solution_quality(best_solution, best_blocks, architecture)
        
        self.stats.update({
            'optimization_time': optimization_time,
            'total_iterations': total_iterations,
            'total_acceptances': total_acceptances,
            'total_rejections': total_iterations - total_acceptances,
            'final_quality': final_quality,
            'final_clusters': len(best_solution),
            'quality_improvement': final_quality - initial_quality,
            'cluster_reduction': len(initial_solution) - len(best_solution),
            'acceptance_rate': total_acceptances / total_iterations if total_iterations > 0 else 0
        })
        
        logger.info(f"SA Optimization complete - Final: {len(best_solution)} clusters, quality: {final_quality:.4f}")
        logger.info(f"Quality improvement: {self.stats['quality_improvement']:.4f}, Cluster reduction: {self.stats['cluster_reduction']}")
        
        return best_solution, best_blocks, self.stats
    
    def _calculate_solution_quality(self, solution, blocks, architecture):
        """
        Calculate quality of a solution balancing utilization vs cluster count
        
        Args:
            solution: Set of block IDs in the solution
            blocks: Dictionary of Block objects
            architecture: Architecture object for utilization calculation
            
        Returns:
            float: Quality score (higher is better)
        """
        if not solution:
            return float('-inf')
            
        # Calculate average utilization across all blocks
        total_utilization = 0
        for block_id in solution:
            block = blocks[block_id]
            utilization = architecture.util(block)
            total_utilization += utilization
            
        avg_utilization = total_utilization / len(solution)
        
        # Apply cluster count penalty
        cluster_penalty = (len(solution) / 100.0) * self.cluster_penalty_weight
        
        quality = avg_utilization - cluster_penalty
        
        return quality
    
    def _get_neighbor_solution(self, current_solution, current_blocks, architecture, temperature):
        """
        Generate a neighbor solution by merging blocks
        
        Args:
            current_solution: Current set of block IDs
            current_blocks: Current dictionary of Block objects
            architecture: Architecture object for validation
            temperature: Current temperature for exploration control
            
        Returns:
            tuple: (neighbor_solution, neighbor_blocks) or (None, None) if no valid neighbor
        """
        # Sample blocks from current solution
        sample_size = min(self.neighbor_sample_size, len(current_solution))
        sampled_blocks = random.sample(list(current_solution), sample_size)
        
        # Find all valid merge pairs among sampled blocks
        valid_pairs = self._find_valid_merge_pairs(sampled_blocks, current_blocks, architecture)
        
        if not valid_pairs:
            return None, None
            
        # Get best non-overlapping pairs
        best_pairs = self._get_best_non_overlapping_pairs(valid_pairs)
        
        if not best_pairs:
            return None, None
            
        # Select pair based on temperature
        selected_pair = self._select_pair_with_temperature_bias(best_pairs, temperature)
        
        if selected_pair is None:
            return None, None
            
        # Create neighbor solution by merging the selected pair
        block_a_id, block_b_id = selected_pair
        
        # Create new solution and blocks
        neighbor_solution = set(current_solution)
        neighbor_blocks = deepcopy(current_blocks)
        
        # Remove the two blocks being merged
        neighbor_solution.remove(block_a_id)
        neighbor_solution.remove(block_b_id)
        
        # Create merged block
        merged_block = self._create_merged_block(
            neighbor_blocks[block_a_id], 
            neighbor_blocks[block_b_id]
        )
        
        # Add merged block to solution
        neighbor_solution.add(merged_block.id)
        neighbor_blocks[merged_block.id] = merged_block
        
        # Remove old blocks
        del neighbor_blocks[block_a_id]
        del neighbor_blocks[block_b_id]
        
        return neighbor_solution, neighbor_blocks
    
    def _find_valid_merge_pairs(self, block_ids, blocks, architecture):
        """
        Find all valid merge pairs among given blocks with overlap scores
        
        Args:
            block_ids: List of block IDs to consider
            blocks: Dictionary of Block objects
            architecture: Architecture object for validation
            
        Returns:
            list: List of (block_a_id, block_b_id, overlap_score) tuples
        """
        valid_pairs = []
        
        for i in range(len(block_ids)):
            for j in range(i + 1, len(block_ids)):
                block_a_id = block_ids[i]
                block_b_id = block_ids[j]
                
                block_a = blocks[block_a_id]
                block_b = blocks[block_b_id]
                
                # Check if merge is valid
                if architecture.valid(block_a, block_b):
                    # Calculate overlap score
                    overlap_score = self._get_input_overlap_score(block_a, block_b)
                    valid_pairs.append((block_a_id, block_b_id, overlap_score))
        
        return valid_pairs
    
    def _get_input_overlap_score(self, block_a, block_b):
        """
        Calculate input overlap score using Jaccard similarity
        
        Args:
            block_a, block_b: Block objects
            
        Returns:
            float: Jaccard similarity score
        """
        intersection = block_a.inputs & block_b.inputs
        intersection_size = len(intersection)
        
        if intersection_size == 0:
            return 0
            
        union = block_a.inputs | block_b.inputs
        union_size = len(union)
        
        return intersection_size / union_size
    
    def _get_best_non_overlapping_pairs(self, valid_pairs):
        """
        Select best non-overlapping pairs using greedy approach (like ATLAS)
        
        Args:
            valid_pairs: List of (block_a_id, block_b_id, overlap_score) tuples
            
        Returns:
            list: List of (block_a_id, block_b_id, overlap_score) for selected pairs
        """
        # Sort by overlap score (descending)
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[2], reverse=True)
        
        # Greedy selection of non-overlapping pairs
        selected_pairs = []
        used_blocks = set()  # Use regular set instead of BitMap for block IDs
        
        for block_a_id, block_b_id, overlap_score in sorted_pairs:
            # If either block is already used, skip
            if block_a_id in used_blocks or block_b_id in used_blocks:
                continue
                
            # Add to selected pairs
            selected_pairs.append((block_a_id, block_b_id, overlap_score))
            used_blocks.add(block_a_id)
            used_blocks.add(block_b_id)
        
        return selected_pairs
    
    def _select_pair_with_temperature_bias(self, best_pairs, temperature):
        """
        Select a pair based on temperature for exploration vs exploitation
        
        Args:
            best_pairs: List of (block_a_id, block_b_id, overlap_score) tuples
            temperature: Current temperature
            
        Returns:
            tuple: (block_a_id, block_b_id) or None if no selection
        """
        if not best_pairs:
            return None
            
        if temperature > self.high_temp_threshold:
            # High temperature: explore by selecting from top 50% of pairs
            selection_size = max(1, len(best_pairs) // 2)
            candidate_pairs = best_pairs[:selection_size]
        else:
            # Low temperature: exploit by selecting from top 2-3 pairs
            selection_size = min(3, len(best_pairs))
            candidate_pairs = best_pairs[:selection_size]
        
        # Randomly select from candidates
        selected = random.choice(candidate_pairs)
        return (selected[0], selected[1])
    
    def _create_merged_block(self, block_a, block_b):
        """
        Create a new merged block from two existing blocks
        
        Args:
            block_a, block_b: Block objects to merge
            
        Returns:
            Block: New merged block
        """
        # Generate new unique block ID by simple increment
        new_block_id = self.next_block_id
        self.next_block_id += 1
        
        # Create new block without SNN (we'll set inputs and outputs directly)
        from atlas_base import Block  # Import Block class
        merged_block = Block(new_block_id, [])
        
        # Set outputs and inputs using BitMap operations
        merged_block.outputs = block_a.outputs | block_b.outputs
        merged_block.inputs = block_a.inputs | block_b.inputs
        
        return merged_block
    
    def _accept_move(self, current_quality, neighbor_quality, temperature):
        """
        Decide whether to accept a move using SA acceptance criterion
        
        Args:
            current_quality: Quality of current solution
            neighbor_quality: Quality of neighbor solution
            temperature: Current temperature
            
        Returns:
            bool: True if move should be accepted
        """
        # Always accept better solutions
        if neighbor_quality > current_quality:
            return True
            
        # Accept worse solutions with probability based on temperature
        delta = neighbor_quality - current_quality
        probability = math.exp(delta / temperature)
        
        return random.random() < probability
    
    def get_stats(self):
        """Get optimization statistics"""
        return self.stats.copy()
    
    def print_stats(self):
        """Print comprehensive optimization statistics"""
        print("\n" + "="*60)
        print("SIMULATED ANNEALING OPTIMIZATION STATISTICS")
        print("="*60)
        
        print(f"Optimization Time: {self.stats['optimization_time']:.2f} seconds")
        print(f"Total Iterations: {self.stats['total_iterations']}")
        print(f"Temperature Levels: {self.stats['temperature_levels']}")
        print(f"Acceptance Rate: {self.stats['acceptance_rate']:.2%}")
        
        print("\nSolution Quality:")
        print(f"  Initial Quality: {self.stats['initial_quality']:.4f}")
        print(f"  Final Quality: {self.stats['final_quality']:.4f}")
        print(f"  Best Quality Found: {self.stats['best_quality_found']:.4f}")
        print(f"  Quality Improvement: {self.stats['quality_improvement']:.4f}")
        
        print("\nCluster Reduction:")
        print(f"  Initial Clusters: {self.stats['initial_clusters']}")
        print(f"  Final Clusters: {self.stats['final_clusters']}")
        print(f"  Clusters Reduced: {self.stats['cluster_reduction']}")
        print(f"  Reduction Rate: {self.stats['cluster_reduction']/self.stats['initial_clusters']:.2%}")
        
        print("\nAcceptance Statistics:")
        print(f"  Total Acceptances: {self.stats['total_acceptances']}")
        print(f"  Total Rejections: {self.stats['total_rejections']}")
        
        print("="*60)


def main():
    """Example usage of the SA optimizer with ATLAS"""
    import sys
    sys.path.append('.')
    
    # Import the clean ATLAS implementation
    from atlas_base import ATLAS, OptimizedCrossbar, Block, convert_atlas_to_clusters
    from arch.base import Crossbar
        
    # Try to import reader
    from reader import AdarshaReader

    # Create architecture
    crossbar_size = 1024  # Smaller for demonstration
    arch = OptimizedCrossbar(crossbar_size)
    
    '''
    # Load a real SNN model
    model_name = "LeNet"
    model_path = f'../disco_libs/SNN_db/{model_name}'
    conn_file = model_path + '.connection_info.txt' 
    spk_file = model_path + '.spike_info.txt' 
    
    reader = AdarshaReader()
    model, aer = reader.read(conn_file, spk_file)
    reader.addTraffic(model, aer)
    
    snn = model
    logger.info(f"Loaded {model_name} model with {len(snn.nodes())} neurons and {len(snn.edges())} connections")
        
    # Run initial ATLAS clustering
    logger.info("Running initial ATLAS clustering...")
    atlas = ATLAS(snn, arch)
    
    # Run greedy clustering to a moderate target
    atlas._create_minimum_blocks()
    atlas._build_atlas_graph()
    
    # Run limited greedy phase to create starting solution for SA
    initial_merges = atlas._run_greedy_phase(target=1000)  # Target 1000 clusters
    
    logger.info(f"Initial greedy phase completed with {initial_merges} merges")
    
    # Get current solution and blocks
    initial_solution = atlas.get_current_solution()
    initial_blocks = atlas.blocks.copy()
    
    logger.info(f"Initial solution has {len(initial_solution)} clusters")
    
    # Create SA optimizer
    sa_optimizer = ATLAS_SA_Optimizer(
        initial_temp=50.0,
        final_temp=0.1,
        alpha=0.95,
        iterations_per_temp=600,
        cluster_penalty_weight=15.0,
        neighbor_sample_size=50
    )
    
    # Run SA optimization
    logger.info("Running SA optimization...")
    optimized_solution, optimized_blocks, sa_stats = sa_optimizer.optimize(
        initial_blocks, initial_solution, arch
    )
    
    # Print results
    logger.info("SA Optimization Results:")
    for key, value in sa_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Continue with additional greedy phase if desired
    logger.info("Running additional greedy phase on SA results...")
    
    # Update ATLAS with SA results
    atlas.blocks = optimized_blocks
    atlas.update_graph_from_solution(optimized_solution)
    
    # Run final greedy phase
    final_merges = atlas._run_greedy_phase(max_passes=3)

    # Get solution
    final_solution = atlas.get_current_solution()
    final_blocks = atlas.blocks

    # Convert to Clusters object
    pickle.dump((final_solution, final_blocks, snn), open("intermediate.obj", "wb"))
    '''

    final_solution, final_blocks, snn = pickle.load(open("intermediate.obj", "rb"))

    arch = Crossbar(1024)
    clusters_obj = convert_atlas_to_clusters(final_solution, final_blocks, snn, arch)

    # Validate the clustering
    clusters_obj.validate(snn)


    pickle.dump(clusters_obj, open("solution.obj", "wb"))
    logger.info(f"Final solution after additional greedy passes: {len(clusters_obj.cluster_graph)} clusters")
    #logger.info(f"Total reduction: {len(initial_solution)} -> {len(final_solution)} clusters")


if __name__ == "__main__":
    main()
