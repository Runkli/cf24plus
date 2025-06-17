from pprint import pprint
from collections import defaultdict
import networkx as nx
from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
from pqdict import pqdict
import numpy as np
import os
import multiprocessing
from time import perf_counter
import pickle
from sys import getsizeof, getrefcount
import sys
import gc
from math import floor
import json


from abc import ABC, abstractmethod


class GenericArch(ABC):
    @abstractmethod
    def valid(self, cluster):
        pass

    @abstractmethod
    def util(self, cluster):
        pass


class Crossbar(GenericArch):
    def __init__(self, crossbar_size):
        self.crossbar_size = crossbar_size

    def valid(self, cluster, graph: nx.DiGraph):
        all_inputs = set()
        all_outputs = set()
        for neuron in cluster:
            inputs = graph.predecessors(neuron)
            all_inputs.update(inputs)
            all_outputs.add(neuron)
            all_inputs -= all_outputs

            if len(all_outputs) > self.crossbar_size:
                return False

            if len(all_inputs) > self.crossbar_size:
                return False

        return True

    def util(self, cluster, graph):
        pass


class CapacityCluster(GenericArch):
    def __init__(self, cluster_size):
        self.cluster_size = cluster_size

    def valid(self, cluster, graph: nx.DiGraph):
        if len(cluster.nodes) > self.cluster_size:
            return False
        
        return True

    def util(self, cluster, graph):
        pass

