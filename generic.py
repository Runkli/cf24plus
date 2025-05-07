from abc import ABC, abstractmethod
from copy import deepcopy
import networkx as nx
from utils import getTerminalNodes
from arch.base import GenericArch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GenericCluster(ABC):
    @abstractmethod
    def _cluster(self, graph, arch):
        pass

    def _makeClusters(self, clusters, graph):
        return clusters

    def cluster(self, graph, arch):
        clusters = self._cluster(graph, arch)
        output: Clusters = self._makeClusters(clusters, graph)
        output.arch = arch
        return output


class Clusters:
    def __init__(self):
        self.cluster_graph: nx.DiGraph = nx.DiGraph()
        self.arch: GenericArch = None
        self.node_to_cluster = {}

    def _nodeToClusterGen(self):
        for cluster in self.cluster_graph:
            for node in cluster:
                self.node_to_cluster[node] = cluster

    def getClusterForNode(self, node):
        assert len(self.node_to_cluster), "nodeToClusterGen not called?" 
        assert node in self.node_to_cluster, "Node is not in any cluster" 
        return self.node_to_cluster[node]


    def moveNode(self, node, source_cluster, target_cluster, graph):
        """
        Move a node from source_cluster to target_cluster.
        
        Args:
            node: The node to move
            source_cluster: The source cluster containing the node
            target_cluster: The target cluster to move the node to
            graph: The original graph (needed for validation)
            
        Returns:
            bool: True if the move was successful, False otherwise
        """
        # Verify node is in source cluster
        if node not in source_cluster.nodes:
            print(f"Node {node} not found in source cluster")
            return False
        
        # Check if the move would create a valid target cluster
        temp_target = target_cluster.copy()
        temp_target.add_node(node)
        if not self.arch.valid(temp_target, graph):
            print(f"Moving node {node} would create an invalid target cluster")
            return False
        
        # Move the node
        source_cluster.remove_node(node)
        target_cluster.add_node(node)
        
        # Update node-to-cluster mapping
        self.node_to_cluster[node] = target_cluster
        
        # Rebuild the affected parts of the cluster graph
        # This is simpler than trying to calculate all the incremental changes
        self._rebuildClusterGraph(graph)
        
        return True

    def _rebuildClusterGraph(self, graph):
        """Rebuild the cluster graph from scratch."""
        # Create a new cluster graph
        new_graph = nx.DiGraph()
        
        # Add all cluster nodes
        for cluster in self.node_to_cluster.values():
            if cluster not in new_graph:
                new_graph.add_node(cluster)
        
        # Add all inter-cluster edges
        for node in graph.nodes():
            if node not in self.node_to_cluster:
                continue  # Skip nodes not in any cluster
                
            node_cluster = self.node_to_cluster[node]
            node_traffic = graph.nodes[node].get('traffic', 1.0)
            
            # Process outgoing edges
            for succ in graph.successors(node):
                if succ not in self.node_to_cluster:
                    continue
                    
                succ_cluster = self.node_to_cluster[succ]
                if node_cluster == succ_cluster:
                    continue  # Skip intra-cluster connections
                
                # Update edge in cluster graph
                if new_graph.has_edge(node_cluster, succ_cluster):
                    new_graph[node_cluster][succ_cluster]['traffic'] += node_traffic
                else:
                    new_graph.add_edge(node_cluster, succ_cluster, traffic=node_traffic)
        
        # Replace the old cluster graph
        self.cluster_graph = new_graph


    def getOutputNodes(self, graph, cluster):
        out_nodes = set()
        for node in cluster:
            # get outputs in graph
            for out_node in graph.successors(node):
                if self.node_to_cluster[out_node] != cluster:
                    out_nodes.add(node)
                    break
                

        return out_nodes


    def __len__(self):
        return len(self.cluster_graph)

    def __iter__(self):
        for cluster in self.cluster_graph.nodes:
            yield cluster

    def __repr__(self):
        return "Cluster Graph: " + self.cluster_graph.__str__()

    def _validate_unique_nodes(self, graph_ref):
        sum_val = sum([len(cluster.nodes) for cluster in self.cluster_graph])
        all_nodes = set()
        for cluster in self.cluster_graph:
            all_nodes.update(cluster.nodes)

        if len(all_nodes) == sum_val:
            return 0

        return sum_val - len(all_nodes)


    def _validate_nodes(self, graph_ref):
        graph = deepcopy(graph_ref)

        # check whether cluster graph has all nodes from graph
        for cluster in self.cluster_graph:
            for node in cluster.nodes:
                graph.remove_node(node)

        remaining_nodes = len(graph.nodes)
        return remaining_nodes

    def _validate_edges(self, graph_ref):
        graph = deepcopy(graph_ref)

        neuron_to_cluster = {}
        for cluster in self.cluster_graph:
            for neuron in cluster.nodes:
                neuron_to_cluster[neuron] = cluster

        edge_count = len(graph.edges)
        # check for edges between clusters
        for in_neuron, out_neuron in graph.edges:
            in_cluster = neuron_to_cluster[in_neuron]
            out_cluster = neuron_to_cluster[out_neuron]

            # if mapped to the same cluster, skip
            if in_cluster == out_cluster:
                edge_count -= 1
                continue

            # otherwise, check if cluster graph has edge
            if self.cluster_graph.has_edge(in_cluster, out_cluster):
                edge_count -= 1
                continue

            #print("neurons", in_neuron, out_neuron, "edge", id(in_cluster), id(out_cluster), "not found")

        return edge_count

    def _validate_mapping(self, graph_ref):
        valid = 0

        # count valid clusters
        for cluster in self.cluster_graph:
            valid += self.arch.valid(cluster, graph_ref)

        return len(self.cluster_graph) - valid

    def invalid_clusters(self, graph_ref):
        invalid = []
        for cluster in self.cluster_graph:
            if self.arch.valid(cluster, graph_ref):
                continue
            
            invalid.append(cluster)

        return invalid


    def validate(self, graph_ref):
        ret = self._validate_unique_nodes(graph_ref)
        if ret:
            print(f"{bcolors.WARNING}{ret} duplicate nodes detected in clusters{bcolors.ENDC}")
        else:
            print(f"Duplicate check: {bcolors.OKGREEN}PASS{bcolors.ENDC}")

        ret = self._validate_edges(graph_ref)
        if ret:
            print(f"{bcolors.WARNING}Clusters do not contain {ret} edges from graph{bcolors.ENDC}")
        else:
            print(f"Edge check: {bcolors.OKGREEN}PASS{bcolors.ENDC}")

        ret = self._validate_nodes(graph_ref)
        if ret:
            print(f"{bcolors.WARNING}Clusters do not contain {ret} nodes from graph{bcolors.ENDC}")
        else:
            print(f"Node check: {bcolors.OKGREEN}PASS{bcolors.ENDC}")

        ret = self._validate_mapping(graph_ref)
        
        if ret:
            print(f"{bcolors.WARNING} {ret} clusters not valid mappings{bcolors.ENDC}")
        else:
            print(f"Mapping check: {bcolors.OKGREEN}PASS{bcolors.ENDC}")

class Pass:
    def __init__(self):
        return




