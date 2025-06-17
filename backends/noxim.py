import pickle
import networkx as nx
from copy import deepcopy
import sys

sys.path.append("..")

from generic import GenericCluster, Clusters

# should be related to architecture, for now we assume crossbar only
from arch.base import Crossbar
from backends.generic_backend import Backend

class Noxim(Backend):
    def __init__(self, clusters, fileprefix):
        super().__init__(clusters, fileprefix)

        print(NOXIM_CONFIG_TEMPLATE)

    def _codegen(self):
        print("Calling codegen from class Noxim")


class NoximMesh(Noxim):
    def _codegen(self):
        super()._codegen()
        print("Calling codegen from class Noxim Mesh")
        self.generateTrafficFiles()
        self.generateConfig()


    def generateTrafficFiles(self):
        fileprefix = self.fileprefix
        clusters = self.clusters
        
        # Create mapping from DiGraph objects to integer IDs
        cluster_to_id = {}
        id_to_cluster = {}
        
        # Assign IDs to clusters (0 to n-1)
        for i, cluster_node in enumerate(clusters.cluster_graph.nodes):
            cluster_to_id[cluster_node] = i
            id_to_cluster[i] = cluster_node
        
        print(f"Created mapping for {len(cluster_to_id)} clusters")
        
        # Generate bus config file
        bus_config_path = fileprefix + "_bus_config.txt"
        print(bus_config_path)
        bus_config_list = []
        
        for src_cluster in clusters.cluster_graph.nodes:
            src_id = cluster_to_id[src_cluster]
            dst_clusters = clusters.cluster_graph.successors(src_cluster)
            dst_ids = [str(cluster_to_id[dst]) for dst in dst_clusters]
            dst_txt = " ".join(dst_ids)
            
            if dst_txt == "":
                continue
            
            txt = f"{src_id} {dst_txt}"
            bus_config_list.append(txt)
        
        bus_config_txt = "\n".join(bus_config_list)
        with open(bus_config_path, "w") as f:
            f.write(bus_config_txt)
        
        # Generate connection weights file
        conn_weights_path = fileprefix + "_conn_weights.txt"
        conn_weights_list = []
        print(conn_weights_path)
        
        for (src_cluster, dst_cluster), spk in clusters.conn_weights.items():
            src_id = cluster_to_id[src_cluster]
            dst_id = cluster_to_id[dst_cluster]
            txt = f"{src_id}; {dst_id}; {spk}"
            conn_weights_list.append(txt)
        
        conn_weights_txt = "\n".join(conn_weights_list)
        with open(conn_weights_path, "w") as f:
            f.write(conn_weights_txt)
        
        # Generate traffic file (with count = 1)
        traffic_path = fileprefix + "_traffic.txt"
        traffic_list = []
        print(traffic_path)
        
        aer_traffic = dict(sorted(clusters.aer_traffic.items()))
        for spktime, conns in aer_traffic.items():
            for (src_cluster, dst_cluster), spkcount in conns.items():
                src_id = cluster_to_id[src_cluster]
                dst_id = cluster_to_id[dst_cluster]
                txt = f"{spktime}; {src_id}; {dst_id}; 1"
                traffic_list.append(txt)
        
        traffic_txt = "\n".join(traffic_list)
        with open(traffic_path, "w") as f:
            f.write(traffic_txt)
        
        # Generate full traffic file (with actual spike counts)
        traffic_path = fileprefix + "_traffic_full.txt"
        traffic_list = []
        print(traffic_path)
        
        aer_traffic = dict(sorted(clusters.aer_traffic.items()))
        for spktime, conns in aer_traffic.items():
            for (src_cluster, dst_cluster), spkcount in conns.items():
                src_id = cluster_to_id[src_cluster]
                dst_id = cluster_to_id[dst_cluster]
                txt = f"{spktime}; {src_id}; {dst_id}; {spkcount}"
                traffic_list.append(txt)
        
        traffic_txt = "\n".join(traffic_list)
        with open(traffic_path, "w") as f:
            f.write(traffic_txt)
        
        # Store the mapping for potential use in other methods
        self.cluster_to_id = cluster_to_id
        self.id_to_cluster = id_to_cluster


    def generateConfig(self):
        x = 4
        y = 4
        config_str = NOXIM_CONFIG_TEMPLATE.format(x, y, "test_file")
        print(config_str)
        pass
    



NOXIM_CONFIG_TEMPLATE = \
'''
# NOC & WIRED CONFIGURATION
#
# Topologies:
#   MESH
#   BUTTERFLY
#   BASELINE
#   OMEGA
#
#   BUTTERFLY, BASELINE, and OMEGA are Delta Network topologies
# topology: MESH
# X and Y mesh sizes
topology: MESH
mesh_dim_x: {0} # 4
mesh_dim_y: {1} # 4
n_delta_tiles: 32
# number of flits for each router buffer
buffer_depth: 4
# size of flits, in bits
flit_size: 64
# lenght in mm of router to hub connection
r2h_link_length: 2.0
# lenght in mm of router to router connection
r2r_link_length: 0.1
n_virtual_channels: 1

# Routing algorithms:
#   XY
#   DELTA
#   WEST_FIRST
#   NORTH_LAST
#   NEGATIVE_FIRST
#   ODD_EVEN
#   DYAD
#   TABLE_BASED
# Each of the above labels should match a corresponding
# implementation in the routingAlgorithms source code directory
routing_algorithm: XY
routing_table_filename: ""

# Routing specific parameters
#   dyad_threshold: double
dyad_threshold: 0.6

# Selection Strategies:
#   RANDOM
#   BUFFER_LEVEL
#   NOP
# Each of the above labels should match a corresponding
# implementation in the selectionStrategies source code directory
selection_strategy: RANDOM

#
# WIRELESS CONFIGURATION
#
Hubs:
    defaults:
    # channels from which Hub can receive/transmit
        rx_radio_channels: [0]
        tx_radio_channels: [0]
    # list of node tiles attached to the hub
        attached_nodes: []
    # size of buffers connecting the hub to tiles
        to_tile_buffer_size: 4
        from_tile_buffer_size: 4
    # size of antenna tx/rx
        rx_buffer_size: 4
        tx_buffer_size: 4

# for each hub, the same parameters specified above can be customized
# If not specified, the above default values will be used
# What is usually needed to be customized specifically for each hub is
# the set of nodes that are connected to it. In this simple topology
# we have 4 hubs (0-3) connected to the four nodes of the 2x2
# sub-meshes

    0:
      attached_nodes: [0]

    1:
      attached_nodes: [2]

    2:
      attached_nodes: [8]

    3:
      attached_nodes: [1]

# Transmission channels configuration
# each channel modelizes the transmission over a given frequency that
# can be used by a set of communicating hubs
RadioChannels:
    defaults:
    # data rate in Gb/s affect the number of cycles required for a
    # flit transmission
        data_rate: 16
    # bit error rate (CURRENTLY UNSUPPORTED)
        ber: [0, 0]
    # mac policies:

    # who has the token releas only when a complete packet has
    # been sent
        #[TOKEN_PACKET]

    # who has the token, release only after a fixed number of
    # cycles, even no transmission is occurring
        #[TOKEN_HOLD, num_hold_cycles]

    # who has the token, holds the packet until needed for
    # transmissions, until a max number of cycles is reached
        #[TOKEN_MAX_HOLD, max_hold_cycles]
        mac_policy: [TOKEN_PACKET]


# SIMULATION PARAMETERS
#
clock_period_ps: 12500
# duration of reset signal assertion, expressed in cycles
reset_time: 0
# overal simulation lenght, expressed in cycles
simulation_time: 2400000
# collect stats after a given number of cycles
stats_warm_up_time: 0
# power breakdown, nodes communication details
detailed: false
# stop after a given amount of load has been processed
max_volume_to_be_drained: 0
show_buffer_stats: false

# Winoc
# enable wireless, when false, all wireless channel configuration is
# ignored
use_winoc: false
# experimental power saving strategy
use_wirxsleep: false

# Verbosity level:
#   VERBOSE_OFF
#   VERBOSE_LOW
#   VERBOSE_MEDIUM
#   VERBOSE_HIGH
verbose_mode: VERBOSE_MEDIUM

# Trace
trace_mode: false
trace_filename: "trace_temp.txt"

min_packet_size: 2
max_packet_size: 16
packet_injection_rate: 0.01
probability_of_retransmission: 0.01

# Traffic distribution:
#   TRAFFIC_RANDOM
#   TRAFFIC_TRANSPOSE1
#   TRAFFIC_TRANSPOSE2
#   TRAFFIC_HOTSPOT
#   TRAFFIC_TABLE_BASED
#   TRAFFIC_BIT_REVERSAL
#   TRAFFIC_SHUFFLE
#   TRAFFIC_BUTTERFLY
traffic_distribution: TRAFFIC_TRACE_BASED
# when traffic table based is specified, use the following
# configuration file
traffic_table_filename: ""
traffic_trace_filename: "{2}" # traffic file here
switch_config_filename: ""

switch_delay_ps: 8500
link_delay_ps: 500

num_pes_per_lane: 4
num_bus_lanes: 4
two_sided_pe: true
duo_port_pe: true
num_sw_control_hor: 1
num_sw_control_ver: 1

mpe_ports: 241
neusb_bus_config_filename: "LeNet_512/LeNet_512_bus_config.txt"

output_mode: NORMAL_MODE
# output_mode: DEBUG_MODE
# output_mode: EX_STAT_MODE
# output_mode: NORMAL_MODE
output_filename: "out_tmp.txt"
'''
