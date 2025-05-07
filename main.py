#!/usr/bin/env python3

import networkx as nx
from copy import deepcopy
import sys
from pqdict import pqdict
from functools import partial
import time
import logging

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
        self.outputs = set(output_neurons)  # Neurons at this block's outputs
        self.inputs = set()                 # Combined inputs to these neurons
        
        # Pre-compute inputs if SNN is provided
        if snn is not None:
            for neuron in self.outputs:
                # Get all predecessors (inputs) for this neuron
                predecessors = set(snn.predecessors(neuron))
                self.inputs.update(predecessors)
        
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
        combined_inputs = block_a.inputs.union(block_b.inputs)
        if len(combined_inputs) > self.crossbar_size:
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
    
    def _get_terminal_nodes(self, graph):
        """Get input and output neurons of a graph"""
        root_set = set()
        leaf_set = set()
        
        for node in graph.nodes:
            has_inputs = 0
            has_outputs = 0
            
            for _ in graph._pred[node]:
                has_inputs = 1
                break
                
            for _ in graph._succ[node]:
                has_outputs = 1
                break
                
            if has_inputs == 0:
                root_set.add(node)
            if has_outputs == 0:
                leaf_set.add(node)
                
        return root_set, leaf_set
    
    def _get_input_overlap_score(self, block_a, block_b):
        """
        Calculate the input overlap score between two blocks
        Using Jaccard similarity (intersection over union)
        """
        # Use pre-computed input sets from Block objects
        intersection_size = len(block_a.inputs.intersection(block_b.inputs))
        
        if intersection_size == 0:
            return 0
            
        union_size = len(block_a.inputs) + len(block_b.inputs) - intersection_size
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
        input_to_blocks = {}
        for block_id, block in self.blocks.items():
            for input_neuron in block.inputs:
                input_to_blocks.setdefault(input_neuron, set()).add(block_id)
        
        # Add edges between blocks that share inputs and can be merged
        edges_added = 0
        
        # For each block
        for i in range(num_blocks):
            block_a_id = block_ids[i]
            block_a = self.blocks[block_a_id]
            
            # Print progress every 100 blocks
            if i % 100 == 0:
                logger.info(f"Building ATLAS graph: {i}/{num_blocks} blocks processed")
                
            # Find candidate blocks that share at least one input with this block
            candidates = set()
            for input_neuron in block_a.inputs:
                if input_neuron in input_to_blocks:
                    candidates.update(input_to_blocks[input_neuron])
            
            # Remove current block and already processed blocks
            candidates = {b for b in candidates if b > block_a_id}
            
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
        
        # Create new block without SNN (we'll set inputs directly)
        new_block = Block(new_block_id, block_a.outputs.union(block_b.outputs))
        
        # Set inputs directly
        new_block.inputs = block_a.inputs.union(block_b.inputs)
        
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
        
        # Clear validation cache as it's now invalid
        self.validation_cache.clear()
        
        return new_block_id
    
    def _select_best_merges(self):
        """
        Select the best non-overlapping set of merges for this pass
        Returns a list of (block_a_id, block_b_id) pairs
        """
        # Create dictionary of edges by weight - faster than pqdict
        edge_weights = {}
        
        # Add all edges to the dictionary
        for u, v, data in self.atlas_graph.edges(data=True):
            edge_weights[(u, v)] = data['weight']
        
        # Sort edges by weight (descending)
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Select best non-overlapping edges
        best_merges = []
        merged_nodes = set()
        
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
    
    def cluster(self):
        """Main clustering algorithm"""
        # Create minimum blocks
        self._create_minimum_blocks()
        
        # Build ATLAS graph
        self._build_atlas_graph()
        
        # Perform merging passes until no more merges can be done
        start_time = time.time()
        total_merges = 0
        pass_num = 1
        
        # Initialize stats for progress tracking
        initial_blocks = len(self.blocks)
        
        while True:
            current_blocks = len(self.atlas_graph.nodes)
            progress_pct = (initial_blocks - current_blocks) / initial_blocks * 100
            
            logger.info(f"Pass {pass_num}: Starting with {current_blocks} blocks ({progress_pct:.2f}% reduction)")
            
            pass_start_time = time.time()
            
            # Perform one pass of merging
            merges_done, new_blocks = self._perform_merge_pass()
            total_merges += merges_done
            
            pass_duration = time.time() - pass_start_time
            
            logger.info(f"Pass {pass_num}: Performed {merges_done} merges in {pass_duration:.2f}s, {current_blocks-merges_done} blocks remaining")
            
            # If no merges were done, we're done
            if merges_done == 0:
                break
                
            pass_num += 1
        
        self.times['merging'] = time.time() - start_time
        logger.info(f"Merging complete: {total_merges} total merges in {self.times['merging']:.2f} seconds")
        
        # Convert final blocks to subgraphs
        final_blocks = []
        for block_id in self.atlas_graph.nodes:
            block = self.blocks[block_id]
            
            # Create a subgraph with all neurons (outputs and inputs)
            all_neurons = block.outputs.union(block.inputs)
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


def main():
    """Example usage with a sample SNN"""
    # Create a sample SNN
    snn = nx.DiGraph()
    
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
    
    # Create the optimized crossbar architecture
    crossbar_size = 1024
    arch = OptimizedCrossbar(crossbar_size)
    
    # Create ATLAS instance
    atlas = ATLAS(snn, arch)
    
    # Run clustering
    clusters = atlas.cluster()
    
    # Print results
    logger.info(f"Clustering complete: {len(clusters)} clusters")
    
    # Print performance stats
    stats = atlas.get_performance_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
   main()
