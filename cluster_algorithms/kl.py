#!/bin/python3

import pickle
import networkx as nx
from copy import deepcopy
import sys

sys.path.append("..")

from clustering.generic import GenericCluster, Clusters

# should be related to architecture, for now we assume crossbar only
from arch.base import Crossbar


class KLCluster(GenericCluster):
    # TODO: Require 'traffic' data on the edges of graph
    def _cluster(self, graph, arch: Crossbar):
        # save predecessors for quick lookup
        predecessors = {}

        for n in graph.nodes:
            inputs = set(graph.predecessors(n))
            predecessors[n] = inputs

        clusters = []

        graph_ud = deepcopy(graph).to_undirected()

        # entire graph is one cluster
        # start with all nodes in cluster
        clusters.append(graph_ud)

        final_clusters = []

        while clusters:
            # get one of the clusters
            cluster = clusters.pop()

            # if no edges exist in the current cluster, we can't bisect with KL further, we just divide
            if not cluster.edges:
                if arch.valid(cluster, graph):
                    final_clusters.append(cluster.to_directed())
                else:
                    if len(cluster.nodes) == 1:
                        final_clusters.append(cluster.to_directed())
                        continue

                    nodes_list = list(cluster.nodes)
                    half = len(nodes_list) // 2
                    A = nodes_list[:half]
                    B = nodes_list[half:]
                    sA = graph_ud.subgraph(A)
                    sB = graph_ud.subgraph(B)
                    clusters.append(sA)
                    clusters.append(sB)

                continue

            max_swaps = len(cluster) ** 2
            subclusters = nx.community.kernighan_lin_bisection(cluster,
                                                               weight='traffic',
                                                               max_iter=max_swaps,)

            for subcluster in subclusters:
                subgraph = graph_ud.subgraph(subcluster)
                if arch.valid(subgraph, graph):
                    final_clusters.append(subgraph.to_directed())
                else:
                    clusters.append(subgraph)

        return final_clusters

    def _makeClusters(self, clusters: list[nx.DiGraph], graph: nx.DiGraph):
        clustersObj = Clusters()

        neuron_to_cluster = {}
        for cluster in clusters:
            for neuron in cluster.nodes:
                neuron_to_cluster[neuron] = cluster

        cluster_graph = nx.DiGraph()

        for cluster in clusters:
            for neuron in cluster.nodes:
                # for every neuron in a cluster
                # find neighbors from other clusters
                for in_neuron in graph.predecessors(neuron):
                    other_cluster = neuron_to_cluster[in_neuron]
                    # if other neuron is in the same cluster, skip
                    if cluster == other_cluster:
                        continue

                    neuron_traffic = graph.nodes[in_neuron]['traffic']
                    # if cluster edge exists, add
                    # neuron traffic to existing traffic
                    if cluster_graph.has_edge(other_cluster, cluster):
                        traffic = cluster_graph[other_cluster][cluster]['traffic']
                        new_traffic = traffic + neuron_traffic
                        cluster_graph[other_cluster][cluster]['traffic'] = new_traffic
                    else:
                        cluster_graph.add_edge(other_cluster, cluster, traffic=neuron_traffic)

                for out_neuron in graph.successors(neuron):
                    other_cluster = neuron_to_cluster[out_neuron]
                    # if other neuron is in the same cluster, skip
                    if cluster == other_cluster:
                        continue

                    neuron_traffic = graph.nodes[neuron]['traffic']
                    # if cluster edge exists, add
                    # neuron traffic to existing traffic
                    if cluster_graph.has_edge(cluster, other_cluster):
                        traffic = cluster_graph[cluster][other_cluster]['traffic']
                        new_traffic = traffic + neuron_traffic
                        cluster_graph[cluster][other_cluster]['traffic'] = new_traffic
                    else:
                        cluster_graph.add_edge(cluster, other_cluster, traffic=neuron_traffic)

        clustersObj.cluster_graph = cluster_graph

        return clustersObj





