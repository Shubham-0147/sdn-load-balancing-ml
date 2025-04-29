#!/usr/bin/env python3
"""
sdn_controller.py: Ryu controller for dynamic load balancing in SDN.
Uses LSTM (traffic prediction) and Random Forest (path selection).
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.ofproto import ether
from ryu.lib.packet import packet, ethernet, ipv4
from ryu.topology import event, switches
import numpy as np
from src.utils import load_config, load_models
import networkx as nx
import time

class DynamicLoadBalancer(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    def __init__(self, *args, **kwargs):
        super(DynamicLoadBalancer, self).__init__(*args, **kwargs)
        self.config = load_config("config/config.yaml")
        # Load pre-trained models
        self.lstm_model, self.rf_model = load_models(
            self.config['model']['lstm_model_path'],
            self.config['model']['rf_model_path']
        )
        self.topology_api_app = self
        self.datapaths = {}     # Datapath dictionary: dpid -> datapath object
        self.net = nx.DiGraph() # Network graph
        self.monitor_thread = hub.spawn(self._monitor)  # Start monitoring thread

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Install table-miss flow entry when a switch connects.
        """
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Table-miss flow entry (send unmatched packets to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, 
                                          ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=0,
                                 match=match, instructions=inst)
        datapath.send_msg(mod)
        self.logger.info(f"Default flow installed on switch {datapath.id}")

    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        """
        Event handler for new switches/links in the network topology.
        Builds/updates a networkx graph with nodes and edges (with port info).
        """
        switch_list = switches.Switches(self.topology_api_app)
        switches_list = [s.dp.id for s in switch_list]
        self.net.clear()
        # Add switch nodes
        for sw in switches_list:
            self.net.add_node(sw)
        # Add links (bidirectional with port info)
        link_list = switches.Links(self.topology_api_app)
        for link in link_list:
            src = link.src
            dst = link.dst
            self.net.add_edge(src.dpid, dst.dpid, port=src.port_no)
        self.logger.info(f"Topology updated: Nodes {switches_list}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle incoming packets (table-miss). Performs basic L2 learning (optional)
        or forwards to controller for processing. Could also be extended to detect flows.
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        # Parse packet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether.ETH_TYPE_LLDP:
            # Ignore LLDP packets for topology discovery
            return

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ipv4_pkt:
            return
        dst = ipv4_pkt.dst
        src = ipv4_pkt.src
        in_port = msg.match['in_port']

        self.logger.info(f"Packet in: {src} -> {dst} at switch {datapath.id}")

        # Optionally: base logic for simple forwarding
        # Here we rely on monitoring for directing flows.

    def _monitor(self):
        """
        Periodically poll flow/port stats and perform dynamic routing decisions.
        """
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            self._process_stats()
            hub.sleep(self.config['monitor']['interval'])

    def _request_stats(self, datapath):
        """
        Send flow/port stats request to switches.
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Request port statistics
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _process_stats(self):
        """
        After receiving stats, use ML models to predict and reroute if needed.
        """
        # Example: collect traffic metrics from previous stats (not implemented fully here)
        # Suppose we collected bytes/transmitted for each link in `self.link_stats`
        # Use LSTM to predict future load (placeholder)
        # For demonstration, we'll generate a random traffic prediction
        predicted_load = np.random.rand()

        # Determine source and destination for rerouting (example values)
        src_sw = 1
        dst_sw = 4

        # Find candidate paths
        all_paths = list(nx.all_simple_paths(self.net, src_sw, dst_sw, cutoff=self.config['monitor']['top_n_paths']))
        if not all_paths:
            self.logger.warning("No path found between %s and %s", src_sw, dst_sw)
            return

        # Compute feature vector for each path (e.g., sum predicted load on each link)
        path_features = []
        for path in all_paths:
            load_sum = 0
            for i in range(len(path)-1):
                # In a real scenario, retrieve link load or predicted value for link (path[i] -> path[i+1])
                load_sum += predicted_load  # placeholder: same load for each link
            path_features.append([load_sum])

        # Use RF to select best path index
        # Flatten features if RF was trained on a flat vector
        flat_features = np.array([feat for feat in path_features]).reshape(1, -1)
        try:
            best_index = int(self.rf_model.predict(flat_features)[0])
        except Exception as e:
            self.logger.error(f"RF model prediction failed: {e}")
            best_index = 0

        chosen_path = all_paths[best_index]
        self.logger.info(f"Chosen path: {chosen_path}")

        # Install flow rules along the chosen path
        self._install_path(src_sw, dst_sw, chosen_path)

    def _install_path(self, src_sw, dst_sw, path):
        """
        Install flow entries along the specified path for the pair (src, dst).
        Clears old flows for this src-dst (optional), then adds new flows.
        """
        # Example: remove existing flows for src-sw to dst-sw (not implemented)
        # Iterate through switches in path
        for i in range(len(path)-1):
            curr_sw = path[i]
            next_sw = path[i+1]
            out_port = self.net[curr_sw][next_sw]['port']
            datapath = self.datapaths.get(curr_sw)
            if not datapath:
                continue
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser

            # Match IPv4 packets with given src/dst (modify as needed)
            match = parser.OFPMatch(eth_type=ether.ETH_TYPE_IP,
                                    ipv4_src=src, ipv4_dst=dst)
            actions = [parser.OFPActionOutput(out_port)]
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(datapath=datapath, priority=10,
                                     match=match, instructions=inst)
            datapath.send_msg(mod)
            self.logger.info(f"FlowMod: {curr_sw} -> {next_sw} (out_port {out_port})")
