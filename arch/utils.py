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
import signal
import json

mpl.rcParams['figure.dpi'] = 300

state = {
        "iter": 0,
        "prog": None,
        "op": None,
        "stage": None,
}


class Solution:
    def __init__(self, A, B=None, cost=0):
        self.A: nx.DiGraph = A
        self.B: nx.DiGraph = B
        self.initial_cost = cost
        self.size_root = None
        self.size_leaf = None

    def resulting_edges(self):
        final_edges = set(self.A.edges)
        if self.B:
            final_edges.update(set(self.B.edges))

        return final_edges

    def is_empty(self):
        return len(self.A.edges) == 0 and len(self.B.edges) == 0

    def __repr__(self):
        return f"Solution with A = {hex(id(self.A))} and B={hex(id(self.B))}"

    def __hash__(self):
        #return hash(frozenset({self.A, self.B}))
        #return hash(self.A) * 13 + hash(self.B)
        return hash(self.A) ^ hash(self.B)

    def __eq__(self, other):
        #this = frozenset({self.A, self.B})
        #that = frozenset({other.A, other.B})

        return (self.A == other.A and self.B == other.B) or (self.A == other.B and self.B == other.A)

        #return this == that

class SolutionIdx:
    def __init__(self, A, B=None, cost=0):
        self.A: int = A
        self.B: int = B
        self.initial_cost = cost

    def __repr__(self):
        return f"Solution with A = {self.A} and B={self.B}"

    def __hash__(self):
        #return hash(frozenset({self.A, self.B}))
        #return hash(self.A) * 13 + hash(self.B)
        return hash(self.A) ^ hash(self.B)

    def __eq__(self, other):
        return (self.A == other.A and self.B == other.B) or (self.A == other.B and self.B == other.A)


def generate_related_solutions(DG, subgraph, key_to_subgraph_id, 
                               id_to_graph, xb_size,
                               nodes_cache, costFunc):
    # get possible ways to expand subgraph
    # instead of checking which ones are mergable
    # we know that they have to be ones where we
    # overlaps in the nodes, so why not just
    # expand from input nodes

    this_graph_id = id(subgraph)
    in_nodes = nodes_cache[subgraph]['root']
    out_nodes = nodes_cache[subgraph]['leaf']

    a_edges = nodes_cache[subgraph]['edges']
    a_traffic = nodes_cache[subgraph]['traffic']

    local_solutions = set()
    same_solution = Solution(this_graph_id, this_graph_id)

    a_nodes = nodes_cache[subgraph]['nodes']
    len_in_nodes = len(in_nodes)
    len_out_nodes = len(out_nodes)

    graph_ids_to_check_against = set()

    for i in in_nodes:
        output_nodes = DG.successors(i)
        for d in output_nodes:
            if d in out_nodes:
                continue

            if d not in key_to_subgraph_id:
                continue

            other_graph_id = key_to_subgraph_id[d]

            if other_graph_id in graph_ids_to_check_against:
                continue

            graph_ids_to_check_against.add(other_graph_id)

    for other_graph_id in graph_ids_to_check_against:
        other_graph = id_to_graph[other_graph_id]
        other_in = nodes_cache[other_graph]['root']

        # instead of checking union size,
        # for small graphs, it's likely they will fit totally
        # first check if total nodes is more than xb
        # if it is, then check if we can fit with intersection
        # if total not more than xb size, we can continue normally
        if len_in_nodes + len(other_in) > xb_size:
            in_intersect_len = len(in_nodes.intersection(other_in))
            in_union_len = len_in_nodes + len(other_in) - in_intersect_len

            if in_union_len > xb_size:
                continue

        other_out = nodes_cache[other_graph]['leaf']

        out_intersect_len = len(out_nodes.intersection(other_out))
        out_union_len = len_out_nodes + len(other_out) - out_intersect_len
        if out_union_len > xb_size:
            continue

        b_nodes = nodes_cache[other_graph]['nodes']
        b_edges = nodes_cache[other_graph]['edges']
        b_traffic = nodes_cache[other_graph]['traffic']

        args = (a_nodes, b_nodes)
        #args = (a_edges, b_edges, a_traffic, b_traffic)

        S = SolutionIdx(this_graph_id, other_graph_id)
        S.initial_cost = costFunc(args)
        local_solutions.add(S)

    #local_solutions.remove(same_solution)

    return local_solutions


def get_next_candidates(NN, subgraphs, crossbar_size, costFunc):
    t0 = perf_counter()

    # given subgraph, get input nodes
    nodes_cache = {}

    print("Generating key to subgraph dict")
    state["stage"] = "Generating key to subgraph dict"
    key_to_subgraph_id = {}

    id_to_graph = {}

    traffic_data = nx.get_edge_attributes(NN, 'traffic')

    for s in subgraphs:
        i, o = getTerminalNodes(s)
        nodes_cache[s] = {}
        nodes_cache[s]['root'] = i
        nodes_cache[s]['leaf'] = o
        nodes_cache[s]['nodes'] = set(s.nodes)

        edges = set(s.edges)
        nodes_cache[s]['edges'] = edges
        traffic = [traffic_data[edge[0], edge[1]] for edge in edges]
        nodes_cache[s]['traffic'] = sum(traffic)

        for k in o:
            key_to_subgraph_id[k] = id(s)

        id_to_graph[id(s)] = s

    all_solutions = set()
    rejected_graphs = set()

    print("Generating next solutions")
    state["stage"] = "Generating next solutions"
    c = 1
    l = len(subgraphs)
    for g in subgraphs:
        #print(f"\r{c}/{l}", end='')
        prog_str = f"{c}/{l}"
        print(f"\r{prog_str}", end='')
        state["prog"] = prog_str
        c += 1
        b_solutions = generate_related_solutions(NN, g,
                                                 key_to_subgraph_id,
                                                 id_to_graph,
                                                 crossbar_size,
                                                 nodes_cache,
                                                 costFunc)

        if b_solutions:
            all_solutions.update(b_solutions)
        else:
            rejected_graphs.add(nx.DiGraph(g))

    t1 = perf_counter()
    print(f"Time to generate new solutions: {t1-t0}s")
    return all_solutions, rejected_graphs, id_to_graph


def get_starter_partitions(DG):
    # get all candidates
    # go over candidates and find ones with maximum fan in
    # get nodes at the fan-in, and merge them into graphs
    # remove all the nodes that were selected from the candidate list
    # place graph into candidates list

    # in other words get each node,
    # get all incoming edges,
    # save them into a graph

    DG = deepcopy(DG)

    candidates = []

    for node in DG.nodes():
        if DG.in_edges(node) == 0:
            continue

        node_set = [node]
        node_set += list(DG.predecessors(node))
        edge_set = list(DG.in_edges(node))
        if len(edge_set) == 0:
            continue

        subgraph = DG.edge_subgraph(edge_set)

        candidates.append(subgraph)

    return candidates


def makeGraphFromCARLsim(file_str):
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

    nodes = defaultdict(list)

    # Parse log file

    with open("conn.txt", 'r') as f:
        contents = f.read()

    # Split into neuron groups / connection info / weights

    connInfo = contents.split("###\n")[:-1]

    for idx, inf in enumerate(connInfo):
        connInfo[idx] = inf.split('\n')[:-1]

    # Get each connection from each neuron group pair 

    for line in range(len(connInfo)):
        c = connInfo[line]

        groupInfo = c.pop(0).split(' ')
        gin, gout = groupInfo[0], groupInfo[1]

        for neurons in c:
            tmp = neurons.split(' ')
            nin, nout, weight = tmp[0], tmp[1], float(tmp[2])

            nin_name = f"{gin}:{nin}"
            nout_name = f"{gout}:{nout}"

            nodes[nin_name].append(nout_name)

    # Put all nodes into a DiGraph

    DG = nx.DiGraph()

    for key, values in nodes.items():
        d = key.split(':')[0]
        DG.add_node(key, subset_key=d)
        # Add subset attribute for multipartite visualization
        nx.set_node_attributes(DG, {key: {'subset': d}})

        for v in values:
            d = v.split(':')[0]
            DG.add_node(v, subset_key=d)
            nx.set_node_attributes(DG, {v: {'subset': d}})
            DG.add_edge(key, v, color="black")

    traffic = addTrafficInfo('traffic.txt')

    nodes = DG.nodes()

    # I feel like this following part is a little backward.
    # I'll fix it later lol

    # traffic is defined per node as the number of spikes that runs through each edge
    # all edges with the same input neuron will have the same traffic, because it's
    # the neuron which produces the spike train.
    # Traffic is structured as
    # group_id: [node_0_traffic, node_1_traffic, ..., node_n_traffic]
    for group_id, traffic_array in traffic.items():
        # get group id, and traffic for all the nodes in it
        # we now have to set this traffic information of each node
        # to all the nodes that it connects to

        # get all nodes that part of group "group_id",
        # this could be better done if we set
        # a group attribute to each of the nodes, which we already kind of do.
        group_nodes = list(filter(lambda x: int(x.split(':')[0]) == group_id, nodes))

        # enumerate over all nodes in group
        for idx, source_node in enumerate(group_nodes):
            # get all the nodes that it connects to
            successors = DG.successors(source_node)
            # set the traffic of those weights to the traffic value
            for target_node in successors:
                DG[source_node][target_node]['traffic'] = traffic_array[idx]

            nx.set_node_attributes(DG, {source_node: {'traffic': traffic_array[idx]}})

    aer = getAERInfo("aer.txt")

    return DG, aer


def addTrafficInfo(traffic_filename):

    # Parse log file
    with open(traffic_filename, 'r') as f:
        contents = f.read()

    connInfo = contents.split("###\n")[:-1]

    traffic = {}

    for group in connInfo:
        group = group.split('\n')[:-1]
        gid = int(group[0])

        traffics = group[1:]

        traffics = [traffic.strip() for traffic in traffics]

        traffic_matrix = [[int(i) for i in traffic.split()] for traffic in traffics]

        traffic_np = np.array(traffic_matrix)
        traffic_np = np.sum(traffic_np, axis=0)

        traffic[gid] = traffic_np

    return traffic


def getAERInfo(traffic_filename):

    # Parse log file
    with open(traffic_filename, 'r') as f:
        contents = f.read()

    connInfo = contents.split("###\n")[:-1]

    aer = {}

    for group in connInfo:
        # get rid of last newline-caused-space
        group = group.split('\n')[:-1]
        gid = int(group[0])
        aer_times = [int(a) for a in group[1].split()]
        aer_ids = [f"{gid}:{a}" for a in group[2].split()]
        assert len(aer_times) == len(aer_ids)

        for idx, aer_time in zip(aer_ids, aer_times):
            if idx in aer:
                aer[idx].add(aer_time)
            else:
                aer[idx] = set([aer_time])

    return aer


def reverseAER(aer):
    r_view = {}
    for ts, edges in aer.items():
        for edge, count in edges.items():
            ts_c = (ts, count)
            r_view.setdefault(edge, set()).add(ts_c)

    return r_view


def makeGraphFromAdarsha(path, model_name):
    file_path = os.path.join(path, model_name)
    conn_txt = f"{file_path}.connection_info.txt"
    spk_txt = f"{file_path}.spike_info.txt"
    print(conn_txt)
    print(spk_txt)

    graph = nx.DiGraph()

    with open(conn_txt, "r") as f:
        conn_info = f.read()

    with open(spk_txt, "r") as f:
        spk_info = f.read()

    conn_info = conn_info.strip().split('\n')
    spk_info = spk_info.strip().split('\n')

    for connections in conn_info:
        neurons = connections.split()
        neurons = [int(n) for n in neurons]

        src = neurons[0]
        dst = neurons[1:]

        edges = [(src, d) for d in dst]

        graph.add_edges_from(edges)

    for spks in spk_info:
        spikes = spks.split()
        spikes = [int(n) for n in spikes]

        src = spikes[0]
        aer = spikes[1:]
        total_traffic = len(aer)

        if src not in graph.nodes:
            continue

        for edge in graph.out_edges(src):
            graph[edge[0]][edge[1]]['traffic'] = total_traffic

        nx.set_node_attributes(graph, {src: {'traffic': total_traffic}})

    to_remove = set()
    for src, dst, data in graph.edges(data=True):
        if 'traffic' not in data:
            to_remove.add((src, dst))

    # Remove edges that have no traffic
    for edge in to_remove:
        graph.remove_edge(*edge)

    aer = getAERfromAdarsha(spk_txt)
    return graph, aer


def getAERfromAdarsha(file_path):
    with open(file_path, "r") as f:
        aer_info = f.read()

    aer_info = aer_info.strip().split('\n')

    aer = {}

    for spks in aer_info:
        spks = [int(s) for s in spks.split(' ')[:-1]]
        src = spks[0]
        times = spks[1:]

        aer[src] = times

    return aer


def makeCrossBarGraph(crossbar_size=4):
    '''
    Generates a graph with $crossbar_size "in" and "out" nodes,
    connecting them all to one another
    '''

    DG = nx.DiGraph()
    in_ports = ['in'+str(i) for i in range(crossbar_size)]
    out_ports = ['out'+str(i) for i in range(crossbar_size)]

    for i in in_ports:
        for o in out_ports:
            DG.add_edge(i, o)

    return DG


# Get input and output nodes of a graph
def getTerminalNodes(DG):
    DG_root_set = set()
    DG_leaf_set = set()

    for node in DG.nodes:
        has_inputs = 0
        has_outputs = 0

        for _ in DG._pred[node]:
            has_inputs = 1
            break

        for _ in DG._succ[node]:
            has_outputs = 1
            break

        if has_inputs == 0:
            DG_root_set.add(node)
        if has_outputs == 0:
            DG_leaf_set.add(node)

    return DG_root_set, DG_leaf_set


# TODO: Make this faster please
def visualizeGraph(DG, path, highlight=None, mult=1, heat=0, node_size=10):
    '''
    DG: the graph to be visualized
    path: path to save the final image
    highlight: takes in another graph, and highlights it in the main DG graph
    highlight must exist inside DG
    mult: uses multipartite layout for layer by layer visualization
    heat: generates a heatmap based on traffic of each edge
    '''

    DG = deepcopy(DG)
    
    # color everything black
    for edge in DG.edges():
        edge_data = DG.get_edge_data(*edge)
        edge_data['color'] = 'black'


    # if any edges need to be highlighted, color them
    if highlight:
        all_edges = deepcopy(highlight.edges())
        for edge in all_edges:
            edge_data = DG.get_edge_data(*edge)
            edge_data['color'] = 'red'

    # select layout, multipartite is for layer by layer
    pos = nx.spring_layout(DG, weight="traffic",k=5)
    if mult:
        pos = nx.multipartite_layout(DG)

    # get colors
    edge_colors = [DG.edges[edge]['color'] for edge in DG.edges]

    # heatmap coloring
    if heat:
        min_traffic = min(d["traffic"] for u, v, d in DG.edges(data=True))
        max_traffic = max(d["traffic"] for u, v, d in DG.edges(data=True))
        for u, v, d in DG.edges(data=True):
            d["traffic_norm"] = (d["traffic"] - min_traffic) / (max_traffic - min_traffic)

        cmap = plt.cm.get_cmap("coolwarm")

    plt.clf()

    nx.draw_networkx_nodes(DG, pos, node_color="orange", node_size=30)
    nx.draw_networkx_labels(DG, pos)
    #nx.draw(DG, pos, node_color="orange", with_labels=True, node_size=10)
    nx.draw_networkx_edges(DG, pos, edgelist=DG.edges, edge_color="black", width=1, connectionstyle="arc3,rad=0.1")

    # draw highlighted edges again so we draw over everything else
    if highlight:
        all_edges = highlight.edges()
        edge_color = 'red'
        nx.draw_networkx_edges(DG, pos, edgelist=all_edges, edge_color=edge_color, width=2, )

    # select color from predefined colro or heatmap color palette
    if heat:
        for u, v, d in DG.edges(data=True):
            if heat:
                edge_color = cmap(d['traffic_norm'])
            else:
                edge_color = d['color']

            nx.draw_networkx_edges(DG, pos, edgelist=[(u, v)], edge_color=edge_color, width=2)

    plt.savefig(path)

    return


def limit_neuron_fan(DG, limit, direction):
    assert direction in ['in', 'out']
    # get all nodes
    nodes = DG.nodes()

    get_edges = DG.in_edges if direction == 'in' else DG.out_edges 
    c = 0

    for node in nodes:
        # for each node, get the in-edges
        in_edges = get_edges(node, data=True)
        if in_edges:
            # find out how many needs to be removed
            # if remove_n is negative, then that means we have
            # less edges than the limit, which is good
            remove_n = max(0, len(in_edges) - limit)

            # sort the in edges by traffic
            in_edges = sorted(in_edges, key=lambda x: x[2]['traffic'])

            # remove all the edges
            for i in range(remove_n):
                c += 1
                DG.remove_edge(in_edges[i][0], in_edges[i][1])

    return DG


def remove_duplicates(graphs):
    unique = [v1 for i, v1 in enumerate(graphs)
              if not any(nx.utils.graphs_equal(v1, v2) for v2 in graphs[:i])]
    return unique


def prune_double_edges(graph):
    graph = nx.DiGraph(graph)
    edges = list(graph.edges())
    to_remove = []
    for edge in edges:
        reverse = edge[::-1]
        if reverse in edges:
            edge_score = graph.get_edge_data(*edge)['traffic']
            reverse_score = graph.get_edge_data(*reverse)['traffic']
            if edge_score > reverse_score:
                to_remove.append(reverse)
            else:
                to_remove.append(edge)

    for r in to_remove:
        try:
            graph.remove_edge(*r)
        except:
            continue

    return graph


# this function will remove nodes that have no input
# that are not the input layer
# it will also prune nodes that have no output
# that are not the output layer
def prune_weird_nodes(graph):
    old_graph = deepcopy(graph)

    depths = nx.get_node_attributes(graph, "subset")
    max_depth = max(depths.values())
    min_depth = min(depths.values())

    new_graph = deepcopy(old_graph)

    while (True):
        to_remove = []
        old_graph = deepcopy(new_graph)

        for node in new_graph.nodes:
            if new_graph.in_degree(node) == 0 and \
               new_graph.nodes[node]['subset'] != min_depth:
                to_remove.append(node)

            if new_graph.out_degree(node) == 0 and \
               new_graph.nodes[node]['subset'] != max_depth:
                to_remove.append(node)

        for node in to_remove:
            new_graph.remove_node(node)

        # if no more changes, then we're good
        if nx.utils.graphs_equal(old_graph, new_graph):
            break

    return new_graph


def get_best_matches(NN, solution_idxs, all_graphs, costFunc, id_to_graph):

    print("Do setup for get best matches")

    #all_graphs = set(all_graphs)
    all_graphs = set()
    for s in solution_idxs:
        all_graphs.add(s.A)
        all_graphs.add(s.B)

    idx = 0

    ''' TODO: Rewrite the explanation for this, it's outdated
    1. sort with cost function in heap
    2. get best match
    3. remove best match graphs from consideration
    4. repeat function
    '''
    # Combine thread-local dictionaries into one large edge_to_owner dictionary
    solution_idxs = {solution: solution.initial_cost for solution in solution_idxs}

    solution_idxs = pqdict(solution_idxs, reverse=True)
    total_len = len(solution_idxs)

    print("Getting best matches")

    in_solution = set()

    '''
    1. Get best solution
    2a. Get solution.A, mark as in solution
    2b. Get solution.B, mark as in solution
    What about everything else that contains either A or B
    Let's say there is another solution such as (A,C)
    We will skip checking this solution, because A is already in a solution
    What will happen to C?
    If it shows up in another solution such as (C,D) then it's in a solution,
    therefore in the "in_solution" set
    If it doesn't show up in another solution, then we "reject" the solution,
    and they go to the next round.
    This is better because we will be working with the solution objects
    instead of edges, making our code much faster (in theory).
    '''

    final_solution_pairs = set()

    while (solution_idxs):
        current_len = len(solution_idxs)
        prog_str =  f"> Progress: {total_len-current_len+1}/{total_len} {(1-((current_len-1)/total_len)) * 100:.2f}%" 
        print(f"\r{prog_str}", end='')
        state["prog"] = prog_str

        # get best solution
        best_solution = solution_idxs.pop()

        # get best match
        if best_solution.A in in_solution or\
           best_solution.B in in_solution:
            continue

        in_solution.add(best_solution.A)
        in_solution.add(best_solution.B)

        final_solution_pairs.add(best_solution)

        idx += 1

    left_out_graphs = all_graphs - in_solution

    subgraphs = []
    edge_sets = []

    for s_id in final_solution_pairs:
        edges = set(id_to_graph[s_id.A].edges)
        edges.update(set(id_to_graph[s_id.B].edges))

        edge_sets.append(edges)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        subgraphs.append(G)

    for g_id in left_out_graphs:
        g = id_to_graph[g_id]
        edges = set(g.edges)
        edge_sets.append(edges)
        subgraphs.append(g)

    if len(edge_sets) > 1:
        assert set.intersection(*edge_sets) == set(), "There should be no\
                                                   overlap between \
                                                   generated edge sets,\
                                                   something messed up"

    print("finish..")
    del solution_idxs

    return subgraphs


def vis_partials(NN, partial_matches, prefix, rm=0, mult=1):
    print("Visualizing graphs")
    NN = deepcopy(NN)
    partial_matches = deepcopy(partial_matches)

    for idx, graph in enumerate(partial_matches):
        path = f"{prefix}{idx}.png"
        print(graph)
        i, o = getTerminalNodes(graph)
        print(len(i), len(o))

        visualizeGraph(NN, path, highlight=graph, heat=0, mult=mult)
        if rm:
            for edge in graph.edges():
                NN.remove_edge(*edge)


def test_unique_edges(NN, partial_matches):
    # if all edges are unique, we should be
    # able to remove all of them from NN
    # with no exceptions thrown
    NN = deepcopy(NN)
    partial_matches = deepcopy(partial_matches)

    for idx, graph in enumerate(partial_matches):
        for edge in graph.edges():
            NN.remove_edge(*edge)

    print("Edges remaining after removing solutions:", len(NN.edges))

    return NN


def check_overlap(list1, list2):
    for i in list1:
        for j in list2:
            if i == j:
                continue

            ins = set(i.edges).intersection(set(j.edges))
            if ins:
                print(i, j)
                print("wtf")
                exit(1)


def fuse_collapse(NN, partial_matches, XB, crossbar_size, costFunc, checkpoint_name, op):
    assert op in ["merge", "pack"], f"{op} not recognized, must be either merge or pack"

    print_str =\
        f'''
        --------------------------------------------
        | Generating/fusing solutions with {op}    |
        --------------------------------------------
        '''

    print(print_str)

    idx = 0

    final_graphs_file_name = "objs/tmp_final_graphs.obj"
    try:
        os.remove(final_graphs_file_name)
    except:
        pass

    #final_graphs_file = open(final_graphs_file_name, "ab")
    '''
    if os.path.isfile(checkpoint_name) and op == "merge":
        idx, partial_matches = pickle.load(open(checkpoint_name, 'rb'))
    '''

    NN_og = deepcopy(NN)

    state["op"] = op
    while partial_matches:
        state["iter"] = idx
        #print("Generating new candidates")

        state["stage"] = "Getting next candidates"

        if op == "merge":
            solutions, rejected_graphs, id_to_graph = get_next_candidates(NN, partial_matches, crossbar_size, costFunc)
        elif op == "pack":
            solutions, rejected_graphs, id_to_graph = pack_partials(NN, partial_matches, crossbar_size, costFunc)

        #all_rejected_graphs.update(rejected_graphs)
        with open(final_graphs_file_name, "ab") as f:
            pickle.dump(rejected_graphs, f)

        del rejected_graphs

        if len(solutions) == 0:
            break

        state["stage"] = "Getting best matches"
        partial_matches = get_best_matches(NN, solutions, partial_matches,
                                           costFunc,
                                           id_to_graph)

        unique_graphs(NN, partial_matches)
        '''
        if idx % 1 == 0 and op == "merge":
            partial_matches = [nx.DiGraph(p) for p in partial_matches]

            to_file = (idx, partial_matches)
            pickle.dump(to_file, open(checkpoint_name, 'wb'))
        '''

        del id_to_graph
        gc.collect()

    partial_matches = [nx.DiGraph(g) for g in partial_matches]

    final_graphs = set()

    final_graphs_file = open(final_graphs_file_name, "rb")
    #os.remove(final_graphs_file_name)

    try:
        while True:
            graphs = pickle.load(final_graphs_file)
            for g in graphs:
                final_graphs.add(g)

    except EOFError:
        pass

    return final_graphs


class Clusters:
    def __init__(self, cluster_graph, id_to_cluster):
        self.cluster_graph = cluster_graph
        self.id_to_cluster = id_to_cluster

        self.aer_traffic = {}
        self.cluster_aer_traffic = {}
        self.conn_weights = {}


def makeClusters(NN, partial_matches, aer):
    print("Making clusters")

    # make clusters
    ClusterGraph = nx.DiGraph()

    partial_matches = [nx.DiGraph(p) for p in partial_matches]

    # denotes which cluster each node will be in
    id_to_graph = {}

    for idx, cluster in enumerate(partial_matches):
        id_to_graph[idx] = cluster

        ClusterGraph.add_node(idx)

    checked = {}

    # spikes at timestep between each cluster
    aer_traffic = {}
    # total number of spikes between clusters
    conn_weights = {}

    # TODO: Make dict for number of spikes that each cluster generates

    cluster_ids = id_to_graph.keys()
    for cluster_idx in cluster_ids:
        cluster_i = id_to_graph[cluster_idx]
        for cluster_jdx in cluster_ids:
            if cluster_idx == cluster_jdx or \
               frozenset({cluster_idx, cluster_jdx}) in checked:
                continue

            cluster_j = id_to_graph[cluster_jdx]

            intersection = nx.intersection(cluster_i, cluster_j)
            if len(intersection.nodes) == 0:
                continue

            # for each node in the intersection
            for node in intersection:
                # get num of incoming edges
                i_in = cluster_i.in_degree(node)
                j_in = cluster_j.in_degree(node)

                # get num of outgoing edges
                i_out = cluster_i.out_degree(node)
                j_out = cluster_j.out_degree(node)

                # if input to i is 1, set the direction
                d = 1 if i_in else 0

                cond_in = (i_in == j_in)
                cond_out = (i_out == j_out)

                cond = (cond_in or cond_out)

                break

            if not cond:
                if d == 1:
                    src_cluster_idx, dst_cluster_idx = cluster_idx, cluster_jdx
                elif d == 0:
                    src_cluster_idx, dst_cluster_idx = cluster_jdx, cluster_idx

                # Get the edge
                relation = (src_cluster_idx, dst_cluster_idx)
                # If edge not already in conn_weights, add it in
                # conn weights should hold the number of spikes
                # between source and destination cluster
                if relation not in conn_weights:
                    conn_weights[relation] = 0

                for node in intersection.nodes:
                    if node not in aer:
                        continue
                    for spike_time in aer[node]:
                        if spike_time in aer_traffic:
                            # at time spike_time, relation had a spike
                            aer_traffic[spike_time][relation] = aer_traffic[spike_time].get(relation, 0) + 1
                        else:
                            aer_traffic[spike_time] = {}
                            aer_traffic[spike_time][relation] = 1

                        conn_weights[relation] += 1

                ClusterGraph.add_edge(src_cluster_idx, dst_cluster_idx)

            checked[frozenset({cluster_idx, cluster_jdx})] = 1

    clusters = Clusters(ClusterGraph, id_to_graph)
    clusters.aer_traffic = aer_traffic
    clusters.conn_weights = conn_weights

    return clusters


def generateKhanhFiles(clusters: Clusters, fileprefix: str):
    print("Generating files for khanh")
    bus_config_path = fileprefix + "_bus_config.txt"
    print(bus_config_path)
    bus_config_list = []
    for src in clusters.cluster_graph.nodes:
        dst = clusters.cluster_graph.successors(src)
        dst_txt = " ".join([str(d) for d in dst])
        if dst_txt == "":
            continue

        txt = f"{src} {dst_txt}"
        bus_config_list.append(txt)

    bus_config_txt = "\n".join(bus_config_list)
    with open(bus_config_path, "w") as f:
        f.write(bus_config_txt)

    conn_weights_path = fileprefix + "_conn_weights.txt"
    conn_weights_list = []
    for (src, dst), spk in clusters.conn_weights.items():
        txt = f"{src}; {dst}; {spk}"
        conn_weights_list.append(txt)

    conn_weights_txt = "\n".join(conn_weights_list)
    with open(conn_weights_path, "w") as f:
        f.write(conn_weights_txt)

    traffic_path = fileprefix + "_traffic.txt"
    traffic_list = []

    aer_traffic = dict(sorted(clusters.aer_traffic.items()))
    for spktime, conns in aer_traffic.items():
        for (src, dst), spkcount in conns.items():
            txt = f"{spktime}; {src}; {dst}; 1" #{spkcount}"
            traffic_list.append(txt)

    traffic_txt = "\n".join(traffic_list)
    with open(traffic_path, "w") as f:
        f.write(traffic_txt)
    
    traffic_path = fileprefix + "_traffic_full.txt"
    traffic_list = []

    aer_traffic = dict(sorted(clusters.aer_traffic.items()))
    for spktime, conns in aer_traffic.items():
        for (src, dst), spkcount in conns.items():
            txt = f"{spktime}; {src}; {dst}; {spkcount}"
            traffic_list.append(txt)

    traffic_txt = "\n".join(traffic_list)
    with open(traffic_path, "w") as f:
        f.write(traffic_txt)



def visualizeClusters(clusters, heat=1, name="clusters.png"):
    print("Visualizing clusters")
    # visualize clusters
    ClusterGraph = clusters.cluster_graph

    plt.clf()
    nx.draw(ClusterGraph, arrows=True)
    visualizeGraph(ClusterGraph, path="heat"+name, heat=heat, mult=0, node_size=200)


def fusion_conditions(m_root_set, m_leaf_set, n_root_set, n_leaf_set,
                      size, op):


    if op == "merge":
        rl_is_1 = not m_root_set.isdisjoint(n_leaf_set)
        rl_is_2 = not n_root_set.isdisjoint(m_leaf_set)

        if rl_is_1 or rl_is_2:
            return False

    root_disjoint = m_root_set.isdisjoint(n_root_set)
    leaf_disjoint = m_leaf_set.isdisjoint(n_leaf_set)

    # if there's no intersect. at the roots or the leaves, skip
    # if roots are disjoint and leafs are disjoint, skip
    if op == "merge":
        if root_disjoint and leaf_disjoint:
            return False

    # if combined roots are larger than crossbar size, skip
    len_root_union = len(m_root_set.union(n_root_set))
    if len_root_union > size:
        return False

    len_leaf_union = len(m_leaf_set.union(n_leaf_set))
    # if combined leaves are larger than crossbar size, skip
    if len_leaf_union > size:
        return False

    return True


def calculate_root_leaf_sets(partial_matches, ml=0):
    root_leaf_cache = {}

    for idx in range(len(partial_matches)):
        p = partial_matches[idx]
        if len(p.edges) == 0:
            continue

        root_set = set()
        leaf_set = set()

        for node in p.nodes:
            in_d = len(p._pred[node])
            out_d = len(p._succ[node])

            if in_d == 0:
                root_set.add(node)

            if out_d == 0:
                leaf_set.add(node)

        pid = id(p)
        root_leaf_cache[pid] = {}
        root_leaf_cache[pid]['root'] = root_set
        root_leaf_cache[pid]['leaf'] = leaf_set
        p_nodes = set(p.nodes)
        root_leaf_cache[pid]['all_nodes'] = p_nodes
        if ml:
            root_leaf_cache[pid]['multilayer'] = bool(p_nodes - root_set.union(leaf_set))

    return root_leaf_cache


# let's try to rewrite this
def pack_partials(NN, subgraphs, size, costFunc):
    root_leaf_cache = calculate_root_leaf_sets(subgraphs, ml=1)

    id_to_graph = {}

    all_indices = list(range(len(subgraphs)))
    
    nodes_cache = {}
    traffic_data = nx.get_edge_attributes(NN, 'traffic')

    for s in subgraphs:
        i, o = getTerminalNodes(s)
        nodes_cache[s] = {}
        nodes_cache[s]['root'] = i
        nodes_cache[s]['leaf'] = o
        nodes_cache[s]['nodes'] = set(s.nodes)

        edges = set(s.edges)
        nodes_cache[s]['edges'] = edges
        traffic = [traffic_data[edge[0], edge[1]] for edge in edges]
        nodes_cache[s]['traffic'] = sum(traffic)

        id_to_graph[id(s)] = s

    idx_to_id = {}
    for i in all_indices:
        idx_to_id[i] = id(subgraphs[i])

    for p in subgraphs:
        id_to_graph[id(p)] = p

    all_ids = set(id_to_graph.keys())

    print("\rGenerate mergeable solutions              ")

    current = 0
    total_len = len(subgraphs)

    solutions = set()
    in_solution = set()

    connectivity = {}

    # Calculate if fusable
    for midx in range(len(all_indices)):
        print(f"\r> Progress: {current}/{total_len} {(current/total_len) * 100:.2f}%", end='')
        current += 1

        lhsid = idx_to_id[midx]

        m_root_set = root_leaf_cache[lhsid]['root']
        m_leaf_set = root_leaf_cache[lhsid]['leaf']

        m_multi_layer = root_leaf_cache[lhsid]['multilayer']
        if m_multi_layer:
            continue

        for nidx in range(midx+1, len(all_indices)):
            rhsid = idx_to_id[nidx]

            n_root_set = root_leaf_cache[rhsid]['root']
            n_leaf_set = root_leaf_cache[rhsid]['leaf']

            # if combined roots are larger than crossbar size, skip
            len_root_union = len(m_root_set.union(n_root_set))
            if len_root_union > size:
                continue

            len_leaf_union = len(m_leaf_set.union(n_leaf_set))
            # if combined leaves are larger than crossbar size, skip
            if len_leaf_union > size:
                continue

            n_multi_layer = root_leaf_cache[rhsid]['multilayer']

            if n_multi_layer:
                continue

            A = subgraphs[midx]
            B = subgraphs[nidx]

            S = SolutionIdx(lhsid, rhsid)

            a_nodes = nodes_cache[A]['nodes']
            b_nodes = nodes_cache[B]['nodes']

            a_edges = nodes_cache[A]['edges']
            b_edges = nodes_cache[B]['edges']

            a_traffic = nodes_cache[A]['traffic']
            b_traffic = nodes_cache[B]['traffic']

            args = (a_nodes, b_nodes)
            #args = (a_edges, b_edges, a_traffic, b_traffic)
            S.initial_cost = costFunc(args)

            solutions.add(S)

            in_solution.add(lhsid)
            in_solution.add(rhsid)

    rejected_indices = all_ids - in_solution

    rejected_graphs = [nx.DiGraph(id_to_graph[i]) for i in rejected_indices]

    return set(solutions), set(rejected_graphs), id_to_graph


def subsetassignment(NN):
    roots, leafs = getTerminalNodes(NN)

    depths = {}
    dview = {}

    for i in roots:
        depth = nx.shortest_path_length(NN, i)
        for k, v in depth.items():
            dview.setdefault(v, set()).add(k)
            depths[k] = v

    return depths, dview


def prune_per_layer(NN, frac):
    attr = nx.get_node_attributes(NN, "traffic")
    depths, dview = subsetassignment(NN)

    for n in NN.nodes:
        if n not in attr:
            attr[n] = 0

    depths, dview = subsetassignment(NN)
    old_size = len(NN.nodes)

    for layer, neurons in dview.items():
        neurons = sorted(neurons, key=lambda x: attr[x])
        to_prune = int(frac * len(neurons))
        for i in range(to_prune):
            try:
                NN.remove_node(neurons[i])
            except:
                pass

    new_size = len(NN.nodes)
    print(f"Pruned {old_size - new_size} nodes")
    return NN


def prune_heavy_neurons(NN, frac):
    lens = [NN.in_degree(n) for n in NN.nodes]

    avg = sum(lens) / len(lens)

    for n in NN.nodes:
        if NN.in_degree(n) > avg:
            in_edges = NN.in_edges(n, data=True)
            in_edges = sorted(in_edges, key=lambda x: x[2]['traffic'])

            remove_n = floor(len(in_edges) * frac)

            # remove all the edges
            for i in range(remove_n):
                NN.remove_edge(in_edges[i][0], in_edges[i][1])


def prune_all_neurons(NN, frac):
    for n in NN.nodes:
        in_edges = NN.in_edges(n, data=True)
        in_edges = sorted(in_edges, key=lambda x: x[2]['traffic'])

        remove_n = floor(len(in_edges) * frac)

        # remove all the edges
        for i in range(remove_n):
            NN.remove_edge(in_edges[i][0], in_edges[i][1])


def calculate_OIR(graphs):
    total = 0
    for graph in graphs:
        i, o = getTerminalNodes(graph)
        frac = len(o)/len(i)
        total += frac

    total /= len(graphs)
    print(f"Average OIR: {total:.2f}")


def calculate_xb_util(graphs, crossbar_size):
    total_xb_size = crossbar_size * 2
    total = 0
    for graph in graphs:
        edges = len(graph.nodes)
        frac = edges/total_xb_size
        total += frac

    total /= len(graphs)
    print(f"Avg crossbar util: {total*100:.2f}%")


def unique_graphs(NN, graphs):
    NN = deepcopy(NN)
    new_graphs = set()
    for g in graphs:
        gp = NN.subgraph(g)

        if len(gp.edges) == 0:
            continue

        for edge in g.edges:
            NN.remove_edge(*edge)

        new_graphs.add(g)

    return new_graphs


def prune_little_traffic(graph, prune_frac=0.1):
    edges = nx.get_edge_attributes(graph, "traffic")
    print("Pruning little traffic")

    e0 = len(graph.edges)

    print(f"# Edges: {e0}")

    edges = sorted(edges, key=lambda x: edges[x])

    num_to_prune = floor(len(graph.edges) * prune_frac)
    for i in range(num_to_prune):
        graph.remove_edge(*edges[i])

    e1 = len(graph.edges)

    print(f"# Edges: {e1}")

    print(f"Removed {e0-e1} edges")


def signal_status(signum, frame):
    print("Logging:")
    print(state["stage"])
    print(state["prog"])
    sys.stdout.flush()


def is_mappable(NN, cluster, crossbar_size):
    '''
    How do we define whether a cluster is mappable?

    Enough capacity:
    1. Number of input and output neurons is less than crossbar size
    2. If there are intermediate neurons, those have to mapped as well, so there should be enough space both on input and output side

    Topology:
    1. All inputs to a neuron from the NN is inside the cluster

    '''
    problems = []

    i, o = getTerminalNodes(cluster)
    if len(i) > crossbar_size or len(i) > crossbar_size:
        problems.append("I/O capacity error")

    neuron_set = set(cluster.nodes)

    to_check = neuron_set - i
    for neuron in to_check:
        inputs = NN.predecessors(neuron)
        for i in inputs:
            if i not in neuron_set:
                problems.append(
                f"Input {i} to neuron {neuron} is not in cluster"
                )

    problems = "\n".join(problems)

    return problems


def neuron_is_valid(graph_pred_view, cluster_pred_view, neuron):
    '''
    Check if neuron has all its required inputs in the cluster
    or not

    graph_pred_view: predecessor view in the graph
    cluster_pred_view: predecessors view in the cluster

    inputs_in_graph: all required inputs in the SNN
    inputs_in_cluster: inputs placed in the current cluster
    neuron: the neuron we are checking the inputs for
    '''
    inputs_in_graph = graph_pred_view[neuron].keys()
    inputs_in_cluster = cluster_pred_view[neuron].keys()

    return inputs_in_graph == inputs_in_cluster


def check_cluster_mappability_percentage(NN, neurons_to_check, cluster):
    '''
    Given a cluster, it checks the number of neurons
    that are valid

    '''
    NN_pred = NN.pred
    cluster_pred = cluster.pred
    total_neurons = len(neurons_to_check)

    c = 0  # the number of neurons that are valid
    for neuron in neurons_to_check:
        if neuron_is_valid(NN_pred, cluster_pred, neuron):
            c += 1

    return c / total_neurons


def mappability_topology(NN, clusters, includes_in=0):
    '''
    the mapping generated from our method will have all the inputs included because our clusters are mapped as edges -> includes_in == 1

    the KL mapping only clusters neurons, hence all the neurons in the cluster should be neurons for the output of the crossbar (mappedon the output). hence, clusters that only include neurons and no edge information do not include the inputs -> includes_in == 0

    '''
    
    total_edges_p = 0

    avg_p = 0
    for idx, cluster in enumerate(clusters):
        cluster = cluster.to_directed()
        i, o = getTerminalNodes(cluster)

        all_incoming = set()
        for node in cluster.nodes:
            incoming = set(NN.predecessors(node))
            all_incoming.update(incoming)

        #print(f"Cluster {idx}")
        #print(f"# Inputs required for cluster {len(all_incoming)}")

        neurons_to_check = cluster.nodes

        if includes_in:
            neurons_to_check = cluster.nodes - i

        p = check_cluster_mappability_percentage(NN, neurons_to_check, cluster)
        p *= 100
        #print(f'% Mappability: {p:.2f}')

        avg_p += p
        total_edges_p += len(cluster.edges)

        if includes_in:
            inputs = set()
            for n in i:
                inputs.update(NN.predecessors(n))

            total_edges_p -= len(inputs)

    avg_p = avg_p / len(clusters)
    avg_p = round(avg_p, 2)

    total_edges_p = total_edges_p / len(NN.edges) * 100
    total_edges_p = round(total_edges_p, 2)

    print(f'Average mappability of clusters: {avg_p:.2f}%')
    return avg_p, total_edges_p


def mappability_size(NN, clusters, crossbar_size, includes_in=0):
    '''
    try to pack in as many neurons as possible without running over the input limit
    '''

    avg_p = 0

    fractions = []

    edges_mapped = set()
    total_edges_mapped = 0

    for cluster in clusters:
        test_cluster_inputs = set()
        test_cluster_outputs = set()
        num_edges_added = 0
        total_edges = 0

        neurons_to_check = cluster.nodes

        if includes_in:
            i, o = getTerminalNodes(cluster)
            neurons_to_check = cluster.nodes - i

        # sort neurons by their in degree
        neurons_to_check = sorted(neurons_to_check, key=NN.in_degree)

        # start adding as many neurons as you can without going over the input limit
        for neuron in neurons_to_check:
            incoming = set(NN.predecessors(neuron))

            total_edges += len(incoming)
            if len(test_cluster_inputs.union(incoming)) > crossbar_size:
                continue

            if len(test_cluster_outputs) + 1 > crossbar_size:
                continue

            test_cluster_inputs.update(incoming)
            test_cluster_outputs.add(neuron)
            num_edges_added += len(incoming)

        total_edges_mapped += num_edges_added

        fraction = num_edges_added / total_edges
        fraction *= 100

        fractions.append(fraction)

    avg_p = sum(fractions) / len(fractions)
    total_percent = total_edges_mapped / len(NN.edges) * 100

    avg_p = round(avg_p, 2)
    total_percent = round(total_percent, 2)

    print(f'Average mappability of clusters: {avg_p}%')
    print(f'% Mapped edges: {total_percent}')

    return avg_p, total_percent


def valid(cluster, crossbar_size, predecessors):
    all_inputs = set()
    all_outputs = set()
    for neuron in cluster:
        inputs = predecessors[neuron]
        all_inputs.update(inputs)
        all_outputs.add(neuron)

        if len(all_outputs) >= crossbar_size:
            return False

        if len(all_inputs) >= crossbar_size:
            return False

    return True


def g_util(cluster, crossbar_size):
    return len(cluster.edges) / (crossbar_size * crossbar_size)


def g_util_nodes(cluster, crossbar_size):
    i, output = getTerminalNodes(cluster)
    u = len(output) / (crossbar_size)

    return u


def kl_util(cluster, crossbar_size, predecessors):
    n_edges = 0
    for n in cluster.nodes:
        n_edges += len(predecessors[n])

    u = n_edges / (crossbar_size * crossbar_size)
    return u


def kl_util_nodes(cluster, crossbar_size, predecessors):
    u = len(cluster) / (crossbar_size)

    return u


def write_results_dict(filename, args, scores):
    data = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

    # initialize empty dict if it doesn't exist
    data.setdefault(args.experiment_name, {})

    data[args.experiment_name].update(scores)

    with open("kl_results.json", "w") as f:
        json_object = json.dumps(data, indent=2)
        f.write(json_object)


def convert_kl_to_normal(NN, clusters):
    '''
    kl provides nodes only in its clusters,
    we convert this to include edges too
    '''
    new_clusters = []
    for cluster in clusters:
        nodes = set()
        nodes.update(cluster.nodes)

        for n in cluster.nodes:
            pred = NN.predecessors(n)
            nodes.update(pred)
        
        nc = NN.subgraph(nodes)
        nc = nx.DiGraph(nc)
        new_clusters.append(nc)

    return new_clusters


