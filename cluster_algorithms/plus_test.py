#!/bin/python3

import pickle
import networkx as nx
from copy import deepcopy
import sys
import os
import gc
from pqdict import pqdict
from functools import partial
from time import perf_counter
from queue import PriorityQueue

sys.path.append("...utils")
from utils import getTerminalNodes
from itertools import pairwise

from clustering.generic import GenericCluster, Clusters

class Timer:
    def __init__(self):
        self.lap_time = perf_counter()
        pass

    def lap(self, checkpoint=""):
        new_lap_time = perf_counter()
        print(f"Lap time {checkpoint}: {new_lap_time - self.lap_time}s")
        self.lap_time = new_lap_time


# should be related to architecture, for now we assume crossbar only
from arch.base import Crossbar

def input_clusters(graph, arch):
    # special handling
    input_nodes, output_nodes = getTerminalNodes(graph)
    special_clusters = [graph.subgraph(input_nodes)]

    candidates = []
    # random partition the input nodes
    while special_clusters:
        cluster = special_clusters.pop()
        if len(cluster) <= arch.crossbar_size:
            candidates.append(cluster)
            continue

        nodes_list = list(cluster)
        half = len(nodes_list) // 2
        A = nodes_list[:half]
        B = nodes_list[half:]
        special_clusters.append(A)
        special_clusters.append(B)

    return candidates

def get_starter_partitions(DG):
    DG = deepcopy(DG)

    candidates = []

    for node in DG.nodes:
        if DG.in_edges(node) == 0:
            continue

        node_set = set(DG.predecessors(node))
        if len(node_set) == 0:
            continue

        cluster = ClusterI(inputs=node_set, outputs=set([node]))

        candidates.append(cluster)

    return candidates


def collapse_edge(graph, edge, edgeQ, limit=1024):
    # remove nodes of edge from graph
    # make new collapsed node
    # add to graph
    # generate connections
    cluster_a, cluster_b = edge

    '''
    keeping two sets of edges, one in graph representation
    one in priority queue for fast max calculation
    removing node removes edges in networkx graph,
    but doesn't in priority queue, so we may retreive edges
    from priority queue that are no longer real.
    can be fixed by using pqdict, i'll do it later
    '''
    
    neighbor_nodes = []
    for i in nx.neighbors(graph, cluster_a):
        if i == cluster_a or i == cluster_b:
            continue

        neighbor_nodes.append(i)

    for i in nx.neighbors(graph, cluster_b):
        if i == cluster_a or i == cluster_b:
            continue
        
        neighbor_nodes.append(i)
    
    graph.remove_node(cluster_a)
    graph.remove_node(cluster_b)

    # pack union items into new obj
    new_node =  ClusterI(inputs=cluster_a.inputs.union(cluster_b.inputs),
                        outputs=cluster_a.outputs.union(cluster_b.outputs))

    graph.add_node(new_node)

    for neighbor in neighbor_nodes:
        if new_node == neighbor:
            continue

        this_nodes = new_node.inputs | new_node.outputs
        other_nodes = neighbor.inputs | neighbor.outputs
        intersection = this_nodes.intersection(other_nodes)
        len_union = len(this_nodes) + len(other_nodes) - len(intersection)

        # too large to collapse, don't add edge
        if len_union > limit:
            continue

        overlap = len(intersection) / len_union 
        graph.add_edge(new_node, neighbor, overlap=overlap)
        edgeQ.put(Edge(new_node, neighbor, -overlap))

class ClusterI:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

class Edge:
    def __init__(self, a, b, v):
        self.a = a
        self.b = b
        self.val = v

    def __lt__(self, other):
        return self.val < other.val

class PLUSCluster(GenericCluster):
    def _cluster(self, graph, arch: Crossbar):
        t = Timer()

        # save predecessors for quick lookup
        crossbar_size = arch.crossbar_size

        # start with minimal clusters, track their inputs
        starter_clusters = get_starter_partitions(graph)

        t.lap("generate initial clusters")
        # neurons may have the exact same input set
        # and be different neurons

        cluster_graph = nx.Graph()
        for cluster in starter_clusters:
            cluster_graph.add_node(cluster)
        
        # special data structure
        edgeQ = PriorityQueue() 

        # make connections based on inputs of cluster 
        for cluster_a, cluster_b in pairwise(starter_clusters):
            if cluster_a == cluster_b:
                assert False, "clusters can't be equal"
                continue


            # too large to collapse, don't add edge
            this_nodes = cluster_a.inputs | cluster_a.outputs
            other_nodes = cluster_b.inputs | cluster_b.outputs
            intersection = this_nodes.intersection(other_nodes)

            if len(intersection) == 0:
                # if nothing in common, skip
                continue

            len_union = len(this_nodes) + len(other_nodes) - len(intersection)

            # too large to collapse, don't add edge
            if len_union > arch.crossbar_size:
                continue

            overlap = len(intersection) / len_union 

            cluster_graph.add_edge(cluster_a, cluster_b, overlap=overlap)
            edgeQ.put(Edge(cluster_a, cluster_b, -overlap))

        t.lap("make connections")
       
        # generate graph with bitmaps instead
        # fast intersection between clusters
        # collapse graph iteratively
        

        print("iter", end='')
        while not edgeQ.empty():
            print(f"\r{len(cluster_graph.nodes)}", end='')
            #max_edge = max(cluster_graph.edges(data=True), key=lambda x:x[2]['overlap'])
            max_edge = edgeQ.get()
            max_edge = (max_edge.a, max_edge.b)
            if cluster_graph.has_edge(*max_edge):
                collapse_edge(cluster_graph, max_edge, edgeQ, limit=arch.crossbar_size)

        clusters = [frozenset(cluster.outputs) for cluster in cluster_graph]
        inputs = input_clusters(graph, arch) 
        inputs = [frozenset(cluster) for cluster in inputs] 
        clusters += inputs

        return clusters

    def _makeClusters(self, clusters: list[nx.DiGraph], graph: nx.DiGraph):

        clusterobj = Clusters()

        for cluster in clusters:
            clusterobj.cluster_graph.add_node(nx.DiGraph(graph.subgraph(cluster)))


        return clusterobj





