import networkx as nx
import argparse
from os.path import isfile
#from utils import limit_neuron_fan, prune_heavy_neurons, prune_all_neurons
import pickle
from abc import ABC, abstractmethod


class Reader(ABC):
    def read(self, model_path, aer_path):
        model = self._readModel(model_path)
        aer = self._readAER(aer_path)
        return model, aer

    @abstractmethod
    def _readModel(self, model_path):
        pass

    @abstractmethod
    def _readAER(self, aer_path):
        pass

    '''
    given snn graph and spike time information
    add traffic to edges (required by KL)
    '''

    def addTraffic(self, graph, aer):
        nx.set_edge_attributes(graph, 0, 'traffic')

        for neuron, spikes in aer.items():
            spikes = spikes['ts']
            total_traffic = sum(spikes)

            if neuron not in graph.nodes:
                continue

            for edge in graph.out_edges(neuron):
                graph.edges[edge]['traffic'] = total_traffic


class CARLsimReader(Reader):
    def _readModel(self, model_path):
        '''
        Takes path to file containing connection information,
        groupInID groupOutID
        gInNeuron1 gOutNeuron1 weight
        gInNeuron2 gOutNeuron2 weight
        gInNeuron3 gOutNeuron3 weight
        ...        ...         ...
        gInNeuronN gOutNeuronN weight
        ###
        groupInID groupOutID
        ...       ...          ...
        ###
        '''

        # Parse log file
        with open(model_path, 'r') as f:
            contents = f.read()

        # Generate graph
        DG = nx.DiGraph()

        connInfo = contents.split("###\n")[:-1]
        for info in connInfo:
            info = info.split('\n')[:-1]
            grpInfo = info[0]
            connInfo = info[1:]
            _, grpSrc, grpDst = grpInfo.split()

            for conn in connInfo:
                nSrc, nDst, w = conn.split()
                w = float(w)

                src = f'{grpSrc}:{nSrc}'
                dst = f'{grpDst}:{nDst}'
                DG.add_edge(src, dst, weight=w)

        return DG

    def _readAER(self, aer_path):
        # Parse log file
        with open(aer_path, 'r') as f:
            contents = f.read()

        aer = {}
        spkInfo = contents.split("###\n")[:-1]

        for info in spkInfo:
            info = info.split('\n')[:-1]
            smInfo = info[0].split()[1]
            spkInfo = info[1:]
            for spk in spkInfo:
                srcN, ts = spk.split(':')
                srcN = f'{smInfo}:{srcN}'
                ts = ts.split()
                ts = list(map(int, ts))
                n_ts = {srcN: {'ts': ts}}
                aer.update(n_ts)

        return aer


class AdarshaReader(Reader):

    def _readModel(self, model_path):
        graph = nx.DiGraph()

        with open(model_path, "r") as f:
            conn_info = f.read()

        conn_info = conn_info.strip().split('\n')

        for connections in conn_info:
            neurons = connections.split()
            neurons = [int(n) for n in neurons]

            src = neurons[0]
            dst = neurons[1:]

            edges = [(src, d) for d in dst]

            graph.add_edges_from(edges)

        return graph

    def _readAER(self, aer_path):
        with open(aer_path, "r") as f:
            aer_info = f.read()

        aer_info = aer_info.strip().split('\n')

        aer = {}

        for spks in aer_info:
            spks = [int(s) for s in spks.split(' ')[:-1]]
            src = spks[0]
            times = spks[1:]

            aer[src] = {'ts': times}

        return aer


def model_reader(args):
    prep_name = args.prep_name
    prep_aer_name = args.prep_aer_name

    model_name = args.model_name
    crossbar_size = args.crossbar_size
    out_limit = args.out_limit
    heavy_frac = args.heavy_frac
    all_frac = args.all_frac

    if not isfile(prep_name):
        print("Graph file not found, generating graph")
        if model_name == "test":
            NN, AER = makeGraphFromCARLsim('conn.txt')
        else:
            NN, AER = makeGraphFromAdarsha('SNN_db/', model_name)

        print("Original:", NN)

        fan_in_size = int(crossbar_size)
        fan_out_size = fan_in_size

        if out_limit > 0.0:
            fan_out_size = out_limit

        # we take each node and limit the fan-in to $crossbar_size

        NN.remove_nodes_from(list(nx.isolates(NN)))
        NN = limit_neuron_fan(NN, fan_in_size, 'in')
        NN = limit_neuron_fan(NN, fan_out_size, 'out')
        NN.remove_nodes_from(list(nx.isolates(NN)))

        if heavy_frac > 0.0:
            prune_heavy_neurons(NN, heavy_frac)

        if all_frac > 0.0:
            prune_all_neurons(NN, all_frac)

        # prune_little_traffic(NN, 0.6)
        NN.remove_nodes_from(list(nx.isolates(NN)))

        pickle.dump(NN, open(prep_name, "wb"))
        pickle.dump(AER, open(prep_aer_name, "wb"))

    print("Loading model...")
    NN = pickle.load(open(prep_name, "rb"))
    AER = pickle.load(open(prep_aer_name, "rb"))
    print("Loaded graph: ", NN)

    return NN, AER



class ArgReader:
    def __init__(self):
        parser = argparse.ArgumentParser(
                            prog='ProgramName',
                            description='What the program does',)

        parser.add_argument('-p', '--heavy_frac', type=float, default=0.0,
                            help='An optional floating-point argument.')

        parser.add_argument('-a', '--all_frac', type=float, default=0.0,
                            help='An optional floating-point argument.')

        parser.add_argument('-o', '--out_limit', type=int, default=0,
                            help='An optional floating-point argument.')

        parser.add_argument('-n', '--model_name', type=str, default="",
                            help='model name')

        parser.add_argument('-c', '--crossbar_size', type=int, default=0,
                            help='crossbar size')

        parser.add_argument('-e', '--exp_name', type=str, default="",
                            help='experiment name str')

        args = parser.parse_args()

        self.heavy_frac = 0.0
        if args.heavy_frac:
            self.heavy_frac = args.heavy_frac

        self.all_frac = 0.0
        if args.all_frac:
            self.all_frac = args.all_frac

        self.out_limit = 0.0
        if args.out_limit:
            self.out_limit = args.out_limit

        self.crossbar_size = 1024
        if args.crossbar_size:
            self.crossbar_size = args.crossbar_size
        else:
            print("Please specify crossbar size with -c")
            exit(1)

        self.model_name = ''
        if args.model_name == '':
            print("No model name specified with -n")
            exit(1)

        if args.model_name:
            self.model_name = args.model_name

        print(self.model_name)

        if args.exp_name:
            self.experiment_name = args.exp_name
        else:
            self.experiment_name = f"{self.model_name}-c{self.crossbar_size}-h{self.heavy_frac}-a{self.all_frac}"
        
        print(self.experiment_name)

        self.prep_name = f"objs/{self.experiment_name}_prep_graph.obj"
        self.prep_aer_name = f"objs/{self.experiment_name}_prep_graph_aer.obj"
        self.stage1_name = f"objs/{self.experiment_name}_stage1.obj"
        self.stage2_name = f"objs/{self.experiment_name}_stage2.obj"
        self.clusters_name = f"objs/{self.experiment_name}_clusters.obj"
        self.checkpoint_name = f"objs/{self.experiment_name}_checkpoint.obj"
        self.kl_name = f"objs/{self.experiment_name}_kl_clusters.obj"

        self.total_xb_size = self.crossbar_size * self.crossbar_size




