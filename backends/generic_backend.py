import pickle
import networkx as nx
from copy import deepcopy
import sys

sys.path.append("..")

from generic import GenericCluster, Clusters

# should be related to architecture, for now we assume crossbar only
from arch.base import Crossbar
from abc import ABC, abstractmethod

class Backend(ABC):
    def __init__(self, clusters: Clusters, fileprefix: str):
        # validate object
        assert isinstance(clusters, Clusters), f"Backend codegen received {type(clusters)} instead of type Clusters"
        assert isinstance(fileprefix, str), "fileprefix must be of type string"
        self.fileprefix = fileprefix
        self.clusters = clusters

        #clusters.validate()
        self._codegen()

    @abstractmethod
    def _codegen(self):
        pass



