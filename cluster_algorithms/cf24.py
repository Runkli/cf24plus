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
from utils import getTerminalNodes

sys.path.append("..")

from clustering.generic import GenericCluster, Clusters

# should be related to architecture, for now we assume crossbar only
from arch.base import Crossbar


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
    instead of edges, making our code much faster.
    '''

    final_solution_pairs = set()

    while (solution_idxs):
        current_len = len(solution_idxs)
        prog_str =  f"> Progress: {total_len-current_len+1}/{total_len} {(1-((current_len-1)/total_len)) * 100:.2f}%" 
        print(f"\r{prog_str}", end='')

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

    local_solutions = set()

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

        args = (a_nodes, b_nodes)

        S = SolutionIdx(this_graph_id, other_graph_id)
        S.initial_cost = costFunc(args)
        local_solutions.add(S)

    return local_solutions


def get_next_candidates(NN, subgraphs, crossbar_size, costFunc):
    t0 = perf_counter()

    # given subgraph, get input nodes
    nodes_cache = {}

    print("Generating key to subgraph dict")
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
    c = 1

    l = len(subgraphs)
    for g in subgraphs:
        #print(f"\r{c}/{l}", end='')
        prog_str = f"{c}/{l}"
        print(f"\r{prog_str}", end='')
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


def fuse_collapse(NN, partial_matches, crossbar_size, costFunc):

    print_str =\
        f'''
        --------------------------------------------
        | Generating/fusing solutions with     |
        --------------------------------------------
        '''

    print(print_str)

    final_graphs_file_name = "tmp_final_graphs.obj"
    try:
        os.remove(final_graphs_file_name)
    except:
        pass

    while partial_matches:

        solutions, rejected_graphs, id_to_graph = get_next_candidates(NN, partial_matches, crossbar_size, costFunc)

        s = sorted(solutions, key=lambda x: x.initial_cost, reverse=True)

        '''
        for i in s:
            a = id_to_graph[i.A]
            b = id_to_graph[i.B]
            i1, o1 = getTerminalNodes(a)
            i2, o2 = getTerminalNodes(b)

            print(o1, o2, i.initial_cost)
        '''

        print(len(s))

        #all_rejected_graphs.update(rejected_graphs)
        with open(final_graphs_file_name, "ab") as f:
            pickle.dump(rejected_graphs, f)

        del rejected_graphs

        if len(solutions) == 0:
            break

        partial_matches = get_best_matches(NN, solutions, partial_matches,
                                           costFunc,
                                           id_to_graph)

        unique_graphs(NN, partial_matches)

        del id_to_graph
        gc.collect()

    partial_matches = [nx.DiGraph(g) for g in partial_matches]

    final_graphs = set()

    final_graphs_file = open(final_graphs_file_name, "rb")

    try:
        while True:
            graphs = pickle.load(final_graphs_file)
            for g in graphs:
                final_graphs.add(g)

    except EOFError:
        pass

    return final_graphs


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


def cost_overlap(xb_node_size, args):
    a_nodes, b_nodes = args

    '''
    fitness function: how much of the crossbar is being used
    more utilization is better, so this stays the same.
    since two arguments are needed for this,
    we can make use of functools.partial to make a partial function
    before passing this function as "key" to the "max" function
    '''

    if b_nodes is None:
        use = len(a_nodes) / xb_node_size
        score = use
        return score

    node_intersection_len = len(a_nodes.intersection(b_nodes))
    crossbar_util = node_intersection_len / xb_node_size

    score = crossbar_util

    return score


class CF24Cluster(GenericCluster):
    def _cluster(self, graph, arch: Crossbar):
        costFunc = partial(cost_overlap, arch.crossbar_size)

        # save predecessors for quick lookup
        crossbar_size = arch.crossbar_size

        candidates = get_starter_partitions(graph)

        print('------------------------')
        candidates = fuse_collapse(graph, candidates,
                                   crossbar_size, costFunc)

        candidates = unique_graphs(graph, candidates)

        print("Merge finished")

        test_unique_edges(graph, candidates)

        candidates = [nx.DiGraph(i) for i in candidates]
        for candidate in candidates:
            i, o = getTerminalNodes(candidate)
            candidate.remove_nodes_from(i)

        # special handling
        input_nodes, output_nodes = getTerminalNodes(graph)
        special_clusters = [graph.subgraph(input_nodes)]

        while special_clusters:
            cluster = special_clusters.pop()
            if len(cluster) <= arch.crossbar_size:
                candidates.append(cluster)
                continue

            nodes_list = list(cluster.nodes)
            half = len(nodes_list) // 2
            A = nodes_list[:half]
            B = nodes_list[half:]
            sA = nx.DiGraph(graph.subgraph(A))
            sB = nx.DiGraph(graph.subgraph(B))
            special_clusters.append(sA)
            special_clusters.append(sB)

        candidates = [nx.DiGraph(i) for i in candidates]
        #candidates = [i.nodes for i in candidates]

        return candidates

    def _makeClusters(self, clusters: list[nx.DiGraph], graph: nx.DiGraph):

        clustersObj = Clusters()

        neuron_to_cluster = {}
        for cluster in clusters:
            for neuron in cluster.nodes:
                neuron_to_cluster[neuron] = cluster

        cluster_graph = nx.DiGraph()

        for cluster in clusters:
            for neuron in cluster:
                # for every neuron in a cluster
                # find neighbors from other clusters
                for in_neuron in graph.predecessors(neuron):

                    other_cluster = neuron_to_cluster[in_neuron]
                    # if other neuron is in the same cluster, skip
                    if cluster == other_cluster:
                        continue

                    if graph.in_degree(in_neuron) == 0:
                        #print("adding edge", in_neuron, neuron, id(other_cluster), id(cluster))
                        pass

                    neuron_traffic = graph[in_neuron][neuron]['traffic']
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

                    neuron_traffic = graph[neuron][out_neuron]['traffic']
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





