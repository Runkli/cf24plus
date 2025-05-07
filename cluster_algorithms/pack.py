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
from itertools import permutations

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


class ClusterI:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

class EdgeI:
    def __init__(self, a, b, v):
        self.a = a
        self.b = b
        self.v = v

    def __lt__(self, other):
        return self.v < other.v


class PackCluster(GenericCluster):
    def cluster(self, clusters, graph, arch):
        clusters = self._cluster(clusters, graph, arch)
        output: Clusters = self._makeClusters(clusters, graph)
        output.arch = arch
        return output

    def _cluster(self, clusters, graph, arch: Crossbar):
        candidates = []
        t = Timer()
        for cluster in clusters:
            n = len(cluster)
            node = list(cluster)[0]
            c = ClusterI(set(), set())
            for node in cluster:
                c.outputs.add(node)
                c.inputs.update(set(graph.predecessors(node)))

            candidates.append(c)

        crossbar_size = arch.crossbar_size
        final_clusters = []

        cluster_pack = ClusterI(set(), set())
        while candidates:
            print(f"\r{len(candidates)}   ", end='')
            candidate = candidates.pop(0)
            # has space
            if len(cluster_pack.inputs) + len(candidate.inputs) <= arch.crossbar_size and \
                len(cluster_pack.outputs) + len(candidate.outputs) <= arch.crossbar_size:
                cluster_pack.inputs |= candidate.inputs
                cluster_pack.outputs |= candidate.outputs
                continue
            else:
                final_clusters.append(cluster_pack)
                cluster_pack = candidate

        if cluster_pack.inputs or cluster_pack.outputs:
            final_clusters.append(cluster_pack)

        final_clusters = [i.outputs for i in final_clusters]

        return final_clusters

    def _makeClusters(self, clusters: list[nx.DiGraph], graph: nx.DiGraph):
        clustersObj = Clusters()
        for cluster in clusters:
            clustersObj.cluster_graph.add_node(nx.DiGraph(graph.subgraph(cluster)))

        return clustersObj





