#!/usr/bin/env python3
import networkx as nx
from copy import deepcopy
import sys
import math
import random
import time
import logging
import pyroaring
from functools import partial
from pqdict import pqdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Block:
    """
    A Block represents a group of output neurons and their inputs.
    This is the fundamental unit used in the ATLAS algorithm.
    """
    def __init__(self, block_id, output_neurons, snn=None):
        self.id = block_id
        self.outputs = pyroaring.BitMap(output_neurons)  # Neurons at this block's outputs
        self.inputs = pyroaring.BitMap()                 # Combined inputs to these neurons
        
        # Pre-compute inputs if SNN is provided
        if snn is not None:
            for neuron in self.outputs:
                # Get all predecessors (inputs) for this neuron
                predecessors = snn.predecessors(neuron)
                for pred in predecessors:
                    self.inputs.add(pred)
        
    def __repr__(self):
        return f"Block(id={self.id}, outputs={len(self.outputs)}, inputs={len(self.inputs)})"


class OptimizedCrossbar:
    """
    An optimized version of the Crossbar architecture validator.
    Directly checks input and output sizes without building temporary subgraphs.
    """
    def __init__(self, crossbar_size):
        self.crossbar_size = crossbar_size
    
    def valid(self, block_a, block_b):
        """
        Check if merging two blocks would create a valid mapping
        without building a temporary subgraph.
        """
        # Check if combined outputs fit within crossbar
        if len(block_a.outputs) + len(block_b.outputs) > self.crossbar_size:
            return False

        # Check if combined inputs fit within crossbar
        # Use roaring bitmap's fast union operation
        combined_inputs_len = len(block_a.inputs | block_b.inputs)
        if combined_inputs_len > self.crossbar_size:
            return False
        
        return True
    
    def util(self, block):
        """Calculate utilization of a block on the crossbar"""
        total_size = self.crossbar_size * 2  # Rows + Columns
        used_size = len(block.outputs) + len(block.inputs)
        return used_size / total_size


class ATLAS:
    """
    Architecture-aware Topology for Linkage And Synapse mapping
    A graph-based approach for optimized SNN clustering
    Using Block objects to store pre-computed inputs and outputs
    """
    
    def __init__(self, snn, architecture):
        """
        Initialize the ATLAS algorithm
        
        Args:
            snn (nx.DiGraph): The SNN network to cluster
            architecture: Architecture object that provides validity constraints
        """
        self.snn = snn
        self.arch = architecture
        
        # Initialize empty ATLAS graph
        self.atlas_graph = nx.Graph()
        
        # Blocks dictionary - maps block ID to Block object
        self.blocks = {}
        
        # Validation cache to avoid redundant checks
        self.validation_cache = {}
        
        # Final merged blocks
        self.merged_blocks = []
        
        # Initialize timers for performance analysis
        self.times = {
            'initialization': 0,
            'graph_building': 0,
            'merging': 0
        }
        
        # Block ID counter
        self.next_block_id = 0
        
        # Target cluster count for stopping greedy and switching to SA
        self.greedy_target = 500
    
    def _get_input_overlap_score(self, block_a, block_b):
        """
        Calculate the input overlap score between two blocks
        Using Jaccard similarity (intersection over union)
        """
        # Use roaring bitmap's fast intersection and union operations
        intersection = block_a.inputs & block_b.inputs
        intersection_size = len(intersection)
        
        if intersection_size == 0:
            return 0
        
        # Use roaring bitmap's fast union operation    
        union = block_a.inputs | block_b.inputs
        union_size = len(union)
        
        return intersection_size / union_size
    
    def _is_valid_merge(self, block_a_id, block_b_id):
        """Check if merging two blocks would create a valid mapping"""
        # Check cache first
        cache_key = tuple(sorted([block_a_id, block_b_id]))
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Get block objects
        block_a = self.blocks[block_a_id]
        block_b = self.blocks[block_b_id]
        
        # Validate using optimized approach
        result = self.arch.valid(block_a, block_b)
        
        # Cache the result
        self.validation_cache[cache_key] = result
        return result
    
    def _create_minimum_blocks(self):
        """Create the minimum mappable blocks from the SNN"""
        start_time = time.time()
        
        # Get all non-input neurons (neurons that have incoming edges)
        for neuron in self.snn.nodes():
            # Skip if neuron has no incoming edges
            if self.snn.in_degree(neuron) == 0:
                continue
                
            # Create a new block with this single output neuron
            block = Block(self.next_block_id, [neuron], self.snn)
            
            # Add block to our dictionary
            self.blocks[self.next_block_id] = block
            
            # Add to ATLAS graph
            self.atlas_graph.add_node(self.next_block_id)
            
            # Increment block ID counter
            self.next_block_id += 1
        
        self.times['initialization'] = time.time() - start_time
        logger.info(f"Created {len(self.blocks)} minimum blocks in {self.times['initialization']:.2f} seconds")
        
    def _build_atlas_graph(self):
        """Build the ATLAS graph by adding edges between compatible blocks"""
        start_time = time.time()
        
        # Get all block IDs
        block_ids = list(self.blocks.keys())
        num_blocks = len(block_ids)
        
        # Create index of inputs to blocks for faster lookup
        self.input_to_blocks = {}
        for block_id, block in self.blocks.items():
            for input_neuron in block.inputs:
                if input_neuron not in self.input_to_blocks:
                    self.input_to_blocks[input_neuron] = pyroaring.BitMap()
                self.input_to_blocks[input_neuron].add(block_id)
        
        # Add edges between blocks that share inputs and can be merged
        edges_added = 0
        
        # For each block
        for i in range(num_blocks):
            block_a_id = block_ids[i]
            block_a = self.blocks[block_a_id]
            
            # Print progress every 1000 blocks
            if i % 1000 == 0:
                logger.info(f"Building ATLAS graph: {i}/{num_blocks} blocks processed")
                
            # Find candidate blocks that share at least one input with this block using BitMaps
            candidates = pyroaring.BitMap()
            for input_neuron in block_a.inputs:
                if input_neuron in self.input_to_blocks:
                    candidates |= self.input_to_blocks[input_neuron]
            
            # Remove current block and already processed blocks
            candidates -= pyroaring.BitMap(range(block_a_id + 1))
            
            # For each candidate block
            for block_b_id in candidates:
                # Check if merging would be valid
                if self._is_valid_merge(block_a_id, block_b_id):
                    # Calculate overlap score
                    block_b = self.blocks[block_b_id]
                    overlap_score = self._get_input_overlap_score(block_a, block_b)
                    
                    # Add edge with weight based on overlap
                    self.atlas_graph.add_edge(block_a_id, block_b_id, weight=overlap_score)
                    edges_added += 1
        
        self.times['graph_building'] = time.time() - start_time
        logger.info(f"Added {edges_added} edges in {self.times['graph_building']:.2f} seconds")
        
    def _merge_blocks(self, block_a_id, block_b_id):
        """Merge two blocks into a new block"""
        # Get block objects
        block_a = self.blocks[block_a_id]
        block_b = self.blocks[block_b_id]
        
        # Create new block with combined outputs
        new_block_id = self.next_block_id
        self.next_block_id += 1
        
        # Create new block without SNN (we'll set inputs and outputs directly)
        new_block = Block(new_block_id, [])
        
        # Set outputs directly using BitMap operations
        new_block.outputs = block_a.outputs | block_b.outputs
        
        # Set inputs directly using BitMap operations
        new_block.inputs = block_a.inputs | block_b.inputs
        
        # Add new block to our dictionary
        self.blocks[new_block_id] = new_block
        
        # Add new block to ATLAS graph
        self.atlas_graph.add_node(new_block_id)
        
        # Get neighbors of both blocks before removing them
        neighbors_a = set(self.atlas_graph.neighbors(block_a_id))
        neighbors_b = set(self.atlas_graph.neighbors(block_b_id))
        all_neighbors = (neighbors_a.union(neighbors_b)) - {block_a_id, block_b_id}
        
        # Batch process neighbors
        valid_neighbors = []
        for neighbor_id in all_neighbors:
            if self._is_valid_merge(new_block_id, neighbor_id):
                neighbor_block = self.blocks[neighbor_id]
                overlap_score = self._get_input_overlap_score(new_block, neighbor_block)
                valid_neighbors.append((neighbor_id, overlap_score))
        
        # Batch add edges
        for neighbor_id, overlap_score in valid_neighbors:
            self.atlas_graph.add_edge(new_block_id, neighbor_id, weight=overlap_score)
        
        # Remove old blocks from graph
        self.atlas_graph.remove_node(block_a_id)
        self.atlas_graph.remove_node(block_b_id)
        
        # Update input_to_blocks index for the new block
        if hasattr(self, 'input_to_blocks'):
            for input_neuron in new_block.inputs:
                if input_neuron in self.input_to_blocks:
                    self.input_to_blocks[input_neuron].add(new_block_id)
            
            # Remove old blocks from input_to_blocks
            for input_neuron in block_a.inputs:
                if input_neuron in self.input_to_blocks and block_a_id in self.input_to_blocks[input_neuron]:
                    self.input_to_blocks[input_neuron].remove(block_a_id)
            
            for input_neuron in block_b.inputs:
                if input_neuron in self.input_to_blocks and block_b_id in self.input_to_blocks[input_neuron]:
                    self.input_to_blocks[input_neuron].remove(block_b_id)
        
        # invalidate validation cache
        self.validation_cache.clear()
        
        return new_block_id
    
    def _select_best_merges(self):
        """
        Select the best non-overlapping set of merges for this pass
        Returns a list of (block_a_id, block_b_id) pairs
        """
        # Create dictionary of edges by weight 
        edge_weights = {}
        
        # Add all edges to the dictionary
        for u, v, data in self.atlas_graph.edges(data=True):
            edge_weights[(u, v)] = data['weight']
        
        # Sort edges by weight (descending)
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Select best non-overlapping edges
        best_merges = []
        merged_nodes = pyroaring.BitMap()
        
        for edge, weight in sorted_edges:
            u, v = edge
            
            # If either node has already been merged, skip
            if u in merged_nodes or v in merged_nodes:
                continue
                
            # Add to best merges
            best_merges.append((u, v))
            merged_nodes.add(u)
            merged_nodes.add(v)
        
        return best_merges
    
    def _perform_merge_pass(self):
        """Perform one pass of merging, returns number of merges done"""
        # Select best merges for this pass
        best_merges = self._select_best_merges()
        
        if not best_merges:
            return 0, []
            
        # Perform merges
        new_blocks = []
        for block_a_id, block_b_id in best_merges:
            new_block_id = self._merge_blocks(block_a_id, block_b_id)
            new_blocks.append(new_block_id)
        
        return len(best_merges), new_blocks
    
    def _run_greedy_phase(self, target=None, max_passes=None):
        """
        Run a greedy phase with an optional target or maximum passes
        Returns number of merges done
        """
        if target is None:
            target = self.greedy_target
            
        total_merges = 0
        pass_num = 1
        
        while True:
            current_blocks = len(self.atlas_graph.nodes)
            
            # Check if we've reached the target
            if target is not None and current_blocks <= target:
                logger.info(f"Reached target of {target} clusters.")
                break
                
            # Check if we've reached the maximum number of passes
            if max_passes is not None and pass_num > max_passes:
                logger.info(f"Reached maximum number of passes ({max_passes}).")
                break
            
            # Print progress
            logger.info(f"Greedy pass {pass_num}: Starting with {current_blocks} blocks")
            
            # Perform one pass of merging
            merges_done, _ = self._perform_merge_pass()
            total_merges += merges_done
            
            logger.info(f"Greedy pass {pass_num}: Performed {merges_done} merges, {current_blocks-merges_done} blocks remaining")
            
            # If no merges were done, we're done
            if merges_done == 0:
                break
                
            pass_num += 1
        
        return total_merges
        
    def cluster(self):
        """Main clustering algorithm - original greedy approach"""
        # Create minimum blocks
        self._create_minimum_blocks()
        
        # Build ATLAS graph
        self._build_atlas_graph()
        
        # Perform merging passes until no more merges can be done
        start_time = time.time()
        
        # Run greedy phase
        total_merges = self._run_greedy_phase()
        
        self.times['merging'] = time.time() - start_time
        logger.info(f"Greedy phase complete: {total_merges} total merges in {self.times['merging']:.2f} seconds")
        
        # Convert final blocks to subgraphs
        final_blocks = []
        for block_id in self.atlas_graph.nodes:
            block = self.blocks[block_id]
            
            # Create a subgraph with all neurons (outputs and inputs)
            # Convert BitMaps to list for networkx
            all_neurons = list(block.outputs | block.inputs)
            subgraph = self.snn.subgraph(all_neurons).copy()
            
            final_blocks.append(subgraph)
        
        self.merged_blocks = final_blocks
        return final_blocks
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'initialization_time': self.times['initialization'],
            'graph_building_time': self.times['graph_building'],
            'merging_time': self.times['merging'],
            'total_time': sum(self.times.values()),
            'final_blocks': len(self.merged_blocks),
            'average_block_size': sum(len(block.outputs) for block in self.blocks.values()) / len(self.blocks) if self.blocks else 0,
            'average_inputs_per_block': sum(len(block.inputs) for block in self.blocks.values()) / len(self.blocks) if self.blocks else 0
        }


class ATLAS_Alternating(ATLAS):
    """
    ATLAS with alternating Greedy and Simulated Annealing phases
    Inherits most functionality from the original ATLAS
    """
    
    def __init__(self, snn, architecture):
        """Initialize the ATLAS_Alternating algorithm"""
        super().__init__(snn, architecture)
        
        # SA parameters
        self.initial_temp = 100.0
        self.final_temp = 0.1
        self.alpha = 0.95  # Cooling rate
        self.iterations_per_temp = 500
        
        # Cluster penalty weight for quality function
        self.cluster_penalty_weight = 20.0
        
        # Alternating parameters
        self.max_alternating_iterations = 5  # Max times to alternate between SA and greedy
        self.greedy_passes_per_alternation = 3  # Number of greedy passes per alternation
        
        # Additional timers
        self.times['sa_optimization'] = 0
        self.times['alternating_phases'] = 0
    
    def _calculate_solution_quality(self, solution):
        """
        Calculate the quality of a solution
        Higher is better
        """
        # Calculate average utilization
        total_util = 0
        for block_id in solution:
            block = self.blocks[block_id]
            total_util += self.arch.util(block)
        
        avg_util = total_util / len(solution) if solution else 0
        
        # Penalize number of clusters (we want fewer clusters)
        # With high penalty weight to encourage more merging
        num_clusters_penalty = (len(solution) / 100.0) * self.cluster_penalty_weight
        
        # The overall quality is utilization minus penalty
        quality = avg_util - num_clusters_penalty
        
        return quality
    
    def _get_neighbor_solution(self, current_solution):
        """
        Generate a neighbor solution by trying to merge two blocks
        Uses a biased selection towards high-overlap blocks
        Returns:
            new_solution, delta_quality
        """
        # Make a copy of the current solution
        solution_copy = current_solution.copy()
        
        # Select two random blocks to try merging
        if len(solution_copy) < 2:
            return solution_copy, 0
        
        # Instead of purely random selection, bias toward high overlap
        # This helps find better merges faster
        if random.random() < 0.7:  # 70% of the time use bias
            # Find pairs with high overlap
            block_pairs = []
            
            # Limit search to avoid checking all pairs
            sample_size = min(len(solution_copy), 100)
            sample_blocks = random.sample(list(solution_copy), sample_size)
            
            for block_a_id in sample_blocks:
                for block_b_id in sample_blocks:
                    if block_a_id >= block_b_id:
                        continue
                    
                    if self._is_valid_merge(block_a_id, block_b_id):
                        block_a = self.blocks[block_a_id]
                        block_b = self.blocks[block_b_id]
                        overlap = self._get_input_overlap_score(block_a, block_b)
                        block_pairs.append((block_a_id, block_b_id, overlap))
            
            # If we found valid pairs, select one with bias toward higher overlap
            if block_pairs:
                # Sort by overlap (highest first)
                block_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Select from top 10% with higher probability
                top_n = max(1, int(len(block_pairs) * 0.1))
                selected_idx = int(random.triangular(0, len(block_pairs)-1, 0))
                block_a_id, block_b_id, _ = block_pairs[selected_idx]
            else:
                # No valid pairs found, select randomly
                block_a_id, block_b_id = random.sample(list(solution_copy), 2)
        else:
            # 30% of the time, pure random selection for exploration
            block_a_id, block_b_id = random.sample(list(solution_copy), 2)
        
        # Check if they can be merged
        if not self._is_valid_merge(block_a_id, block_b_id):
            return solution_copy, 0
        
        # Calculate current quality
        current_quality = self._calculate_solution_quality(solution_copy)
        
        # Remove the two blocks
        solution_copy.remove(block_a_id)
        solution_copy.remove(block_b_id)
        
        # Merge them - just create a new block, don't update graph since
        # we're only exploring SA solutions
        new_block_id = self.next_block_id
        self.next_block_id += 1
        
        # Create new block without SNN (we'll set inputs and outputs directly)
        block_a = self.blocks[block_a_id]
        block_b = self.blocks[block_b_id]
        new_block = Block(new_block_id, [])
        
        # Set outputs and inputs
        new_block.outputs = block_a.outputs | block_b.outputs
        new_block.inputs = block_a.inputs | block_b.inputs
        
        # Add to blocks dictionary
        self.blocks[new_block_id] = new_block
        
        # Add to solution
        solution_copy.add(new_block_id)
        
        # Calculate new quality
        new_quality = self._calculate_solution_quality(solution_copy)
        
        # Calculate delta quality
        delta_quality = new_quality - current_quality
        
        return solution_copy, delta_quality
    
    def _run_sa_phase(self, initial_solution, starting_temp=None):
        """
        Run a single SA phase
        Returns best solution found and whether any improvement was made
        """
        start_time = time.time()
        
        # Use provided solution as starting point
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        current_quality = self._calculate_solution_quality(current_solution)
        best_quality = current_quality
        initial_quality = current_quality
        
        logger.info(f"Starting SA phase with {len(current_solution)} blocks, quality: {current_quality:.4f}")
        
        # Initialize temperature
        temp = starting_temp if starting_temp is not None else self.initial_temp
        
        # Cooling loop
        iteration = 0
        
        while temp > self.final_temp:
            accepted_moves = 0
            improved_moves = 0
            
            # Iterations at this temperature
            for i in range(self.iterations_per_temp):
                # Get a neighbor solution
                neighbor_solution, delta_quality = self._get_neighbor_solution(current_solution)
                
                # Accept the move if it improves quality or with probability
                # based on temperature and quality delta
                if delta_quality > 0 or random.random() < math.exp(delta_quality / temp):
                    current_solution = neighbor_solution
                    current_quality += delta_quality
                    accepted_moves += 1
                    
                    if delta_quality > 0:
                        improved_moves += 1
                    
                    # Update best solution if needed
                    if current_quality > best_quality:
                        best_solution = current_solution.copy()
                        best_quality = current_quality
                
                iteration += 1
                
                # Progress output every 1000 iterations
                if iteration % 1000 == 0:
                    logger.info(f"SA iteration {iteration}, Temp: {temp:.2f}, "
                                f"Accepted: {accepted_moves}, Improved: {improved_moves}, "
                                f"Current quality: {current_quality:.4f}, "
                                f"Best quality: {best_quality:.4f}, "
                                f"Clusters: {len(current_solution)}")
            
            # Cool down temperature
            temp *= self.alpha
            
            # Output current state
            logger.info(f"Temperature: {temp:.2f}, "
                        f"Accepted moves: {accepted_moves}/{self.iterations_per_temp}, "
                        f"Improved moves: {improved_moves}, "
                        f"Current quality: {current_quality:.4f}, "
                        f"Best quality: {best_quality:.4f}, "
                        f"Clusters: {len(current_solution)}")
            
            # If very few moves were accepted, decrease temp faster
            if accepted_moves < self.iterations_per_temp * 0.05:
                temp *= self.alpha  # Cool down faster
            
            # Break if we didn't accept any moves
            if accepted_moves == 0:
                logger.info("Stopping SA: No moves accepted in this temperature step")
                break
        
        phase_time = time.time() - start_time
        self.times['sa_optimization'] += phase_time
        
        # Check if any improvement was made
        made_improvement = best_quality > initial_quality
        
        logger.info(f"SA phase completed in {phase_time:.2f} seconds. "
                    f"Made improvement: {made_improvement}, "
                    f"Initial quality: {initial_quality:.4f}, "
                    f"Best quality: {best_quality:.4f}, "
                    f"Reduced from {len(initial_solution)} to {len(best_solution)} clusters")
        
        return best_solution, made_improvement
    
    def _update_atlas_graph(self, solution):
        """
        Update the ATLAS graph to match the given solution
        Rebuilds edges between nodes in the solution
        """
        # First clear the existing graph
        self.atlas_graph = nx.Graph()
        
        # Add the nodes from the solution
        for block_id in solution:
            self.atlas_graph.add_node(block_id)
        
        # Rebuild edges
        # For each pair of nodes in the solution
        nodes = list(solution)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                
                # Check if they can be merged
                if self._is_valid_merge(node_i, node_j):
                    # Calculate overlap score
                    block_i = self.blocks[node_i]
                    block_j = self.blocks[node_j]
                    overlap_score = self._get_input_overlap_score(block_i, block_j)
                    
                    # Add edge with weight based on overlap
                    self.atlas_graph.add_edge(node_i, node_j, weight=overlap_score)
        
        logger.info(f"Updated ATLAS graph with {len(self.atlas_graph.nodes())} nodes and {len(self.atlas_graph.edges())} edges")
        
    def cluster(self):
        """Main clustering algorithm with alternating Greedy and SA phases"""
        # Create minimum blocks
        self._create_minimum_blocks()
        
        # Build ATLAS graph
        self._build_atlas_graph()
        
        # Run initial greedy phase to get to target
        start_time = time.time()
        
        logger.info(f"Starting initial greedy phase to reach target of {self.greedy_target} clusters")
        self._run_greedy_phase()
        
        # Get current solution
        current_solution = set(self.atlas_graph.nodes())
        best_solution = current_solution.copy()
        best_quality = self._calculate_solution_quality(best_solution)
        
        logger.info(f"Initial greedy phase complete with {len(current_solution)} clusters")
        
        # Alternating phases
        alternating_start = time.time()
        
        made_improvement = True
        alternation_count = 0
        
        # Continue as long as we're making improvements and haven't exceeded max iterations
        while made_improvement and alternation_count < self.max_alternating_iterations:
            alternation_count += 1
            logger.info(f"Starting alternation {alternation_count}/{self.max_alternating_iterations}")
            
            # Run SA phase
            logger.info("Running SA phase")
            sa_solution, sa_improved = self._run_sa_phase(current_solution)
            sa_quality = self._calculate_solution_quality(sa_solution)
            
            # Update current solution
            current_solution = sa_solution
            
            # Update best solution if better
            if sa_quality > best_quality:
                best_solution = sa_solution.copy()
                best_quality = sa_quality
            
            # Update ATLAS graph with SA solution
            self._update_atlas_graph(current_solution)
            
            # Run greedy phase (limited number of passes)
            logger.info(f"Running greedy phase with {self.greedy_passes_per_alternation} passes")
            greedy_merges = self._run_greedy_phase(max_passes=self.greedy_passes_per_alternation)
            
            # Get new solution after greedy
            greedy_solution = set(self.atlas_graph.nodes())
            greedy_quality = self._calculate_solution_quality(greedy_solution)
            
            # Update current solution
            current_solution = greedy_solution
            
            # Update best solution if better
            if greedy_quality > best_quality:
                best_solution = greedy_solution.copy()
                best_quality = greedy_quality
            
            # Check if this alternation made any improvement
            made_improvement = sa_improved or greedy_merges > 0
            
            logger.info(f"Alternation {alternation_count} complete: "
                        f"SA improved: {sa_improved}, "
                        f"Greedy merges: {greedy_merges}, "
                        f"Current clusters: {len(current_solution)}, "
                        f"Best quality: {best_quality:.4f}")
        
        self.times['alternating_phases'] = time.time() - alternating_start
        self.times['total'] = time.time() - start_time
        
        # Use the best solution found
        self._update_atlas_graph(best_solution)
        
        # Convert final blocks to subgraphs
        final_blocks = []
        for block_id in self.atlas_graph.nodes:
            block = self.blocks[block_id]
            
            # Create a subgraph with all neurons (outputs and inputs)
            all_neurons = list(block.outputs | block.inputs)
            subgraph = self.snn.subgraph(all_neurons).copy()
            
            final_blocks.append(subgraph)
        
        self.merged_blocks = final_blocks
        logger.info(f"Final solution has {len(final_blocks)} clusters")
        
        return final_blocks
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = super().get_performance_stats()
        stats['sa_optimization_time'] = self.times['sa_optimization']
        stats['alternating_phases_time'] = self.times['alternating_phases']
        stats['total_time'] = self.times.get('total', sum(self.times.values()))
        return stats


def main():
    """Example usage with a sample SNN"""
    try:
        # Try to import reader from the provided code
        from reader import AdarshaReader
        
        # ADARSHA MODELS
        model_name = "LeNet"
        model_path = f'../disco_libs/SNN_db/{model_name}'
        conn_file = model_path + '.connection_info.txt' 
        spk_file = model_path + '.spike_info.txt' 
        
        reader = AdarshaReader()
        model, aer = reader.read(conn_file, spk_file)
        reader.addTraffic(model, aer)
        
        # Use the model as our SNN
        snn = model
        logger.info(f"Loaded {model_name} model with {len(snn.nodes())} neurons and {len(snn.edges())} connections")
    except ImportError:
        # If reader is not available, create a synthetic SNN for testing
        logger.info("Reader not found, creating synthetic SNN for testing")
        snn = nx.DiGraph()
        
        # Create a layered network with more connections
        num_layers = 3
        neurons_per_layer = [1000, 500, 200]  # More realistic layer sizes
        connection_prob = 0.05
        
        # Create neurons for each layer
        neuron_id = 0
        for layer, layer_size in enumerate(neurons_per_layer):
            for i in range(layer_size):
                snn.add_node(neuron_id, layer=layer)
                neuron_id += 1
        
        # Create connections between layers
        for layer in range(len(neurons_per_layer) - 1):
            source_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer]
            target_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer + 1]
            
            for source in source_neurons:
                for target in target_neurons:
                    if random.random() < connection_prob:
                        snn.add_edge(source, target, traffic=random.randint(1, 10))
        
        logger.info(f"Created synthetic SNN with {len(snn.nodes())} neurons and {len(snn.edges())} connections")
    
    # Create the optimized crossbar architecture
    crossbar_size = 1024  # Smaller for the example
    arch = OptimizedCrossbar(crossbar_size)
    
    # Create ATLAS instance with alternating phases
    atlas_alt = ATLAS_Alternating(snn, arch)
    
    # Set parameters
    atlas_alt.greedy_target = 1000  # Target for initial greedy phase
    atlas_alt.initial_temp = 50.0
    atlas_alt.final_temp = 0.1
    atlas_alt.alpha = 0.98
    atlas_alt.iterations_per_temp = 500
    atlas_alt.cluster_penalty_weight = 20.0
    atlas_alt.max_alternating_iterations = 8
    atlas_alt.greedy_passes_per_alternation = 2
    
    # Run clustering
    clusters = atlas_alt.cluster()
    
    # Print results
    logger.info(f"Clustering complete: {len(clusters)} clusters")
    
    # Print performance stats
    stats = atlas_alt.get_performance_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
