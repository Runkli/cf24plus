#!/usr/bin/env python3
import networkx as nx
from copy import deepcopy
import sys
import time
import logging
import pyroaring

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

        # Check if we fit in the crossbar even in the worst case
        summed_inputs_len = len(block_a.inputs) + len(block_b.inputs)
        if summed_inputs_len < self.crossbar_size:
            return True
        
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
        
        # Remove old blocks from dictionary
        del self.blocks[block_a_id]
        del self.blocks[block_b_id]
        
        # Invalidate validation cache
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
        """Main clustering algorithm - greedy approach"""
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
    
    def get_current_solution(self):
        """Get the current solution as a set of block IDs"""
        return set(self.atlas_graph.nodes())
    
    def update_graph_from_solution(self, solution):
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
        
        # Create a layered network
        num_layers = 3
        neurons_per_layer = [1000, 500, 200]
        connection_prob = 0.05
        
        # Create neurons for each layer
        neuron_id = 0
        for layer, layer_size in enumerate(neurons_per_layer):
            for i in range(layer_size):
                snn.add_node(neuron_id, layer=layer)
                neuron_id += 1
        
        # Create connections between layers
        import random
        for layer in range(len(neurons_per_layer) - 1):
            source_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer]
            target_neurons = [n for n, d in snn.nodes(data=True) if d['layer'] == layer + 1]
            
            for source in source_neurons:
                for target in target_neurons:
                    if random.random() < connection_prob:
                        snn.add_edge(source, target, traffic=random.randint(1, 10))
        
        logger.info(f"Created synthetic SNN with {len(snn.nodes())} neurons and {len(snn.edges())} connections")
    
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


def convert_atlas_to_clusters(final_solution, final_blocks, original_snn, arch):
    """
    Convert ATLAS final solution to Clusters object
    
    Args:
        final_solution: Set of block IDs
        final_blocks: Dictionary mapping block_id -> Block objects
        original_snn: Original SNN graph
        arch: Architecture object for validation
        
    Returns:
        Clusters: Clusters object compatible with the generic clustering framework
    """
    from clustering.generic import Clusters
    import networkx as nx
    
    clusters_obj = Clusters()
    clusters_obj.arch = arch
    
    # Step 1: Create cluster DiGraphs containing only block outputs
    cluster_digraphs = []
    neuron_to_cluster = {}
    
    for block_id in final_solution:
        block = final_blocks[block_id]
        
        # Create DiGraph for this cluster containing only output neurons
        cluster_digraph = nx.DiGraph()
        
        # Add output neurons to cluster
        output_neurons = list(block.outputs)  # Convert BitMap to list
        cluster_digraph.add_nodes_from(output_neurons)
        
        # Add edges within this cluster (between output neurons)
        for neuron in output_neurons:
            for successor in original_snn.successors(neuron):
                if successor in output_neurons:  # Both neurons in same cluster
                    # Copy edge with its attributes
                    edge_data = original_snn[neuron][successor]
                    cluster_digraph.add_edge(neuron, successor, **edge_data)
        
        cluster_digraphs.append(cluster_digraph)
        
        # Build neuron-to-cluster mapping
        for neuron in output_neurons:
            neuron_to_cluster[neuron] = cluster_digraph
    
    # Step 2: Handle input nodes (nodes with no incoming edges) with bisection
    def get_terminal_nodes(graph):
        """Get input and output nodes from graph"""
        input_nodes = set()
        output_nodes = set()
        
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                input_nodes.add(node)
            if graph.out_degree(node) == 0:
                output_nodes.add(node)
        
        return input_nodes, output_nodes
    
    # Find input nodes that aren't already in any cluster
    input_nodes, output_nodes = get_terminal_nodes(original_snn)
    unhandled_input_nodes = []
    
    for input_node in input_nodes:
        if input_node not in neuron_to_cluster:
            unhandled_input_nodes.append(input_node)
    
    # Special handling for input nodes - bisect until they fit
    if unhandled_input_nodes:
        special_clusters = [original_snn.subgraph(unhandled_input_nodes)]
        
        while special_clusters:
            cluster = special_clusters.pop()
            
            if len(cluster) <= arch.crossbar_size:
                # Fits in architecture, add to clusters
                cluster_digraph = nx.DiGraph(cluster)
                cluster_digraphs.append(cluster_digraph)
                
                # Update neuron-to-cluster mapping
                for neuron in cluster_digraph.nodes:
                    neuron_to_cluster[neuron] = cluster_digraph
            else:
                # Too big, bisect it
                nodes_list = list(cluster.nodes)
                half = len(nodes_list) // 2
                A = nodes_list[:half]
                B = nodes_list[half:]
                sA = original_snn.subgraph(A)
                sB = original_snn.subgraph(B)
                special_clusters.append(sA)
                special_clusters.append(sB)
    
    # Step 3: Create cluster index mapping (like KL makeClusters)
    cluster_to_idx = {}
    idx_to_cluster = {}
    for idx, cluster in enumerate(cluster_digraphs):
        cluster_to_idx[cluster] = idx
        idx_to_cluster[idx] = cluster
    
    # Step 4: Build inter-cluster traffic graph and generate AER traffic
    cluster_graph = nx.DiGraph()
    aer_traffic = {}
    conn_weights = {}
    
    # Add all clusters as nodes in cluster graph
    for idx in range(len(cluster_digraphs)):
        cluster_graph.add_node(idx)
    
    # Step 5: Generate AER traffic by scanning original SNN edges (like KL approach)
    for neuron in original_snn.nodes():
        if neuron not in neuron_to_cluster:
            continue  # Skip neurons not in any cluster
            
        # Check all successors (outgoing edges)
        for successor in original_snn.successors(neuron):
            if successor not in neuron_to_cluster:
                continue  # Skip if successor not in any cluster
            
            src_cluster = neuron_to_cluster[neuron]
            dst_cluster = neuron_to_cluster[successor]
            
            # Only process inter-cluster edges
            if src_cluster != dst_cluster:
                src_idx = cluster_to_idx[src_cluster]
                dst_idx = cluster_to_idx[dst_cluster]
                
                # Get edge traffic from original SNN
                edge_traffic = original_snn[neuron][successor].get('traffic', 1.0)
                
                # Add edge to cluster graph
                if not cluster_graph.has_edge(src_idx, dst_idx):
                    cluster_graph.add_edge(src_idx, dst_idx, traffic=0)
                
                # Update cluster graph traffic
                cluster_graph[src_idx][dst_idx]['traffic'] += edge_traffic
                
                # Update conn_weights (total spikes between clusters)
                relation = (src_idx, dst_idx)
                conn_weights[relation] = conn_weights.get(relation, 0) + edge_traffic
                
                # Generate AER traffic (distribute traffic across timesteps)
                # For simplicity, distribute evenly across timesteps based on traffic value
                traffic_per_timestep = max(1, int(edge_traffic))
                num_timesteps = max(1, int(edge_traffic / traffic_per_timestep))
                
                for timestep in range(num_timesteps):
                    if timestep not in aer_traffic:
                        aer_traffic[timestep] = {}
                    
                    if relation not in aer_traffic[timestep]:
                        aer_traffic[timestep][relation] = 0
                    
                    aer_traffic[timestep][relation] += traffic_per_timestep
    
    # Step 6: Set up Clusters object (following KL structure)
    clusters_obj.cluster_graph = cluster_graph
    clusters_obj.aer_traffic = aer_traffic
    clusters_obj.conn_weights = conn_weights
    
    # Create id_to_graph mapping (like the original makeClusters)
    clusters_obj.id_to_graph = idx_to_cluster
    
    # Generate node-to-cluster mapping for validation
    clusters_obj.node_to_cluster = {}
    for neuron, cluster in neuron_to_cluster.items():
        cluster_idx = cluster_to_idx[cluster]
        clusters_obj.node_to_cluster[neuron] = cluster_idx
    
    clusters_obj._nodeToClusterGen()  # Generate node-to-cluster mapping
    
    return clusters_obj


# Usage snippet after running ATLAS:
"""
# Run ATLAS clustering
atlas = ATLAS(snn, arch)
clusters = atlas.cluster()

# Get final solution and blocks
final_solution = atlas.get_current_solution()
final_blocks = atlas.blocks

# Convert to Clusters object (now needs architecture parameter)
clusters_obj = convert_atlas_to_clusters(final_solution, final_blocks, snn, arch)

# Validate the clustering
clusters_obj.validate(snn)

# Now you can use clusters_obj with the generic clustering framework
print(f"Number of clusters: {len(clusters_obj)}")
"""

if __name__ == "__main__":
    main()
