# -*- coding: utf-8 -*-
"""
@file    utils.py
@author  Li Yan
@date    2020-05-26
@version 

Defines several utility functions for the environment.
"""

import os, re, sumolib
import lxml.etree as ET
import libsumo as traci

def prep_veh_route(
    net: sumolib.net.Net = None, 
    edge_map: dict = None, 
    charger_route_path: str = None, 
    num_charger: int = 1, 
    depart_step: int = 0, 
    start_edgeIDs: dict = None, 
    charger_path: str = None
):
    """ 
    generate rou.xml file for chargers and chargingStations.xml 
    """
    charger_edge_pos_path = os.path.join(charger_path, 'chargerEdgePos')
    f = open(charger_edge_pos_path, "r")
    cand_charg_stations = [] 
    charg_canalIDs = []
    for line in f:
        items = re.split(",", line)
        edge_pos = (items[0], float(items[1]))
        cand_charg_stations.append(edge_pos)
        charg_canalIDs.append(edge_map[edge_pos[0]][0][2])
    f.close()
    
    # ## create the additional xml for charging stations
    # charger_XML_path = os.path.join(charger_path, 'chargingStations.xml')
    # fw = open(charger_XML_path, "w")
    # fw.write("<additional>\n")
    # charg_canalIDs = []
    # for edge_pos in cand_charg_stations:
        # # print edge_pos[0]+":"+str(net.getEdge(edge_pos[0]).getLength())
        # laneLength = net.getEdge(edge_pos[0]).getLength()
        # startPos = edge_map[edge_pos[0]][0][0]
        # endPos = edge_map[edge_pos[0]][0][1]
        # charg_canalIDs.append(edge_map[edge_pos[0]][0][2])
        # fw.write("\t<chargingStation chargeDelay=\"2\" chargeInTransit=\"1\" chrgpower=\"200000\" efficiency=\"0.95\" endPos=\""+str(endPos)+"\" id=\""+edge_pos[0]+"\" lane=\""+edge_pos[0]+"_0\" startPos=\""+str(startPos)+"\"/>\n")
    # fw.write("</additional>\n")
    # fw.close()

    ## create the initial route for chargers
    if not os.path.exists(charger_route_path):
        fw = open(charger_route_path, "w")
        fw.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        fw.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.sf.net/xsd/routes_file.xsd\">\n")
        fw.write("\t<vType id=\"charger\" accel=\"0.1\" decel=\"10.0\" sigma=\"0.5\" length=\"5\" maxSpeed=\"0.1\" color=\"0,0,1\">\n")
        fw.write("\t\t<param key=\"maximumBatteryCapacity\" value=\"2000\"/>\n")
        fw.write("\t\t<param key=\"actualBatteryCapacity\" value=\"2000\"/>\n")
        fw.write("\t\t<param key=\"vehicleMass\" value=\"1000\"/>\n")
        fw.write("\t\t<param key=\"frontSurfaceArea\" value=\"2\"/>\n")
        fw.write("\t\t<param key=\"airDragCoefficient\" value=\"0.06\"/>\n")
        fw.write("\t\t<param key=\"constantPowerIntake\" value=\"50\"/>\n")
        fw.write("\t\t<param key=\"propulsionEfficiency\" value=\"1.0\"/>\n")
        fw.write("\t\t<param key=\"recuperationEfficiency\" value=\"1.0\"/>\n")
        fw.write("\t</vType>\n")
        for i in range(1,num_charger+1):
            fw.write("\t<vehicle id=\"charger"+str(i)+"\" depart=\""+str(depart_step)+"\" departPos=\"0\" type=\"charger\">\n")
            fw.write("\t\t<route edges=\""+start_edgeIDs["charger"+str(i)][0] +"\"/>\n")
            # startEdgeID = random.choice(all_edges).getID()
            # while self.net.getEdge(startEdgeID).getLength() < 30:
                # startEdgeID = random.choice(all_edges).getID()
            # fw.write("\t\t<route edges=\"" + startEdgeID +"\"/>\n")
            fw.write("\t</vehicle>\n")
        fw.write("</routes>\n")
        fw.close()
    
    return charg_canalIDs
    
def create_sumo_cfg(
    config_path: str = None, 
    net_file: str = None, 
    route_file: str = None, 
    charger_route_file: str = None, 
    label: str = None
):
    """ 
    generate sumo configuration file 
    """
    root = ET.XML('''\
    <configuration>
        <input>
            <net-file value="''' + net_file + '''"/>
            <route-files value="''' + route_file + ''',''' + charger_route_file + '''"/>
            <additional-files value="chargingStations.xml"/>
        </input>
        
        <time>
            <begin value="0"/>
            <end value="-1"/>
            <step-length value="1"/>
        </time>
        
        <routing>
            <routing-algorithm value="astar"/>
            <astar.all-distances value="canal3lane_uturn_dist_mat"/>
        </routing>
        
        <processing>
            <max-depart-delay value="1"/>
            <eager-insert value="false"/>
            <extrapolate-departpos value="false"/>
            <collision.action value="none"/>
            <time-to-teleport value="-1"/>
            <lanechange.overtake-right value="true"/>
            <ignore-route-errors value="false"/>
            <device.rerouting.threads value="4"/>
            <no-internal-links value="false"/>
            <collision.check-junctions value="false"/>
            <threads value="4"/>
        </processing>
        
        <report>
            <no-step-log value="true"/>
            <duration-log.disable value="true"/>
            <duration-log.statistics value="false"/>
            <xml-validation value="never"/>
            <no-warnings value="true"/>
            <verbose value="false"/>
        </report>
    </configuration>
    ''')
    tree = ET.ElementTree(root)
    fw = open(config_path, "wb")
    fw.write(ET.tostring(tree))
    fw.close()
    
def canal_map_gen(
    all_edges: list = None, 
    MAX_SPEED: int = 0, 
    SPEED_TIMES: int = 1
):
    """ 
    partition edges into segments by speed limit 
    """
    canal_map = {}
    tmp_canal_map = {}
    seg_index = 0
    edge_map = {}
    for e in all_edges:
        speed = MAX_SPEED * SPEED_TIMES
        edge_len = e.getLength()
        seg_count = int(edge_len / speed)
        segments = []
        if seg_count > 0:
            for i in range(seg_count):
                if i == seg_count - 1:
                    branches = [ed.getID() for ed in list(e.getOutgoing().keys())]
                    segments.append((i*speed, edge_len, seg_index, branches))
                    tmp_canal_map[seg_index] = (True, branches, e.getID(), i*speed, edge_len)
                else:
                    segments.append((i*speed, (i+1)*speed, seg_index, seg_index + 1))
                    tmp_canal_map[seg_index] = (False, seg_index + 1, e.getID(), i*speed, (i+1)*speed)
                
                seg_index += 1
        else:
            branches = [ed.getID() for ed in list(e.getOutgoing().keys())]
            segments.append((0, edge_len, seg_index, branches))
            tmp_canal_map[seg_index] = (True, branches, e.getID(), 0, edge_len)
            seg_index += 1
            
        edge_map[e.getID()] = segments
        
    for s in range(seg_index):
        segment = tmp_canal_map[s]
        branches = [s, s] # stay or charge actions
        # branches = [s] # only stay action
        if segment[0]:
            for eID in segment[1]:
                branches.append(edge_map[eID][0][2])
        else:
            branches.append(segment[1])
        canal_map[s] = (branches, segment[2], segment[3], segment[4])
        
    return canal_map, edge_map
    
def prep_canal_net(
    net_path: str = None, 
    num_charger: int = 1
):
    """ 
    use sumolib to obtain the network info and edge info 
    """
    net = sumolib.net.readNet(net_path)
    all_edges = net.getEdges()
    chargerIDs = ["charger"+str(c) for c in range(1, num_charger+1)]
    start_edgeIDs = {
        "charger1":('360632526', '360632524#1'), 
        "charger2":('29997526', '-22103865'),
        "charger3":('-29396342', '-195582880#0'),
        "charger4":('-29473626#1', '-15498366')
    }
    return net, all_edges, chargerIDs, start_edgeIDs
    
def prep_veh_depart(
    route_path: str = None, 
    charger_route_path: str = None, 
    MAX_SPEED: int = 0, 
    SPEED_TIMES: int = 1
):
    """ 
    read vessel and charger info from their route XML file
    (string, string, int, int) -> dict, dict dict, int
    """
    veh_depart_time = {}
    veh_info = {}
    veh_orig_speed = {}
    
    ## read vessels' info
    doc = ET.parse(route_path)
    max_veh_num = len(doc.xpath("vehicle"))
    root = doc.getroot()
    for v_type in root.iter('vType'):
        veh_orig_speed[v_type.get('id')] = float(v_type.get('maxSpeed'))
    veh_orig_speed['charger'] = MAX_SPEED * SPEED_TIMES #-1.0
        
    for vehicle in root.findall('vehicle'):
        depart_time = int(vehicle.get('depart'))
        v_type = vehicle.get('type')
        vehicleID = vehicle.get('id')
        for node in vehicle.getiterator():
            if node.tag == 'route':
                routeID = '!' + node.get('id')
                route_edges = node.get('edges').split()
        veh_info[vehicleID] = {
            "type":v_type,
            "depart":depart_time,
            "departLane":vehicle.get('departLane'),
            "departSpeed":vehicle.get('departSpeed'),
            "departPos":vehicle.get('departPos'),
            "routeID":routeID,
            "routeEdges":route_edges
        }
        if depart_time in veh_depart_time:
            veh_depart_time[depart_time].append(vehicleID)
        else:
            veh_depart_time[depart_time] = [vehicleID]
        
    ## read chargers' info
    doc = ET.parse(charger_route_path)
    root = doc.getroot()
    for vehicle in root.findall('vehicle'):
        depart_time = int(vehicle.get('depart'))
        vehicleID = vehicle.get('id')
        if depart_time in veh_depart_time:
            veh_depart_time[depart_time].append(vehicleID)
        else:
            veh_depart_time[depart_time] = [vehicleID]
            
    return veh_depart_time, veh_info, veh_orig_speed, max_veh_num
    
def get_distance(
    canal_map: dict = None,
    o_seg: int = 0,
    d_seg: int = 0,
    # o_edgeID: str = None, 
    # d_edgeID: str = None, 
    # o_pos: float = 0.0, 
    # d_pos: float = 0.0, 
    isDriving: bool = True, 
    INF_DIST: float = 1e30, 
    MAX_DIST: float = 3e6
):
    """ 
    return the distance between two positions
    (string, float, string, float) -> float
    """
    o_edgeID = canal_map[o_seg][1]
    d_edgeID = canal_map[d_seg][1]
    o_pos = canal_map[o_seg][2]
    d_pos = canal_map[d_seg][2]
    dist = float(traci.simulation.getDistanceRoad(
        o_edgeID, o_pos,
        d_edgeID, d_pos,
        isDriving))
        
    ## if origin is downstream of destination, dist is inf
    ## e.g., origin = '-29474883', 200; dest = '-29474883', 0
    if dist > INF_DIST:
        dist = MAX_DIST
        
    return dist
    
def get_route(
    edge_map: dict = None, 
    canal_map: dict = None,
    # o_edgeID: str = None, 
    o_seg: int = 0, 
    # d_edgeID: str = None, 
    d_seg: int = 0
):
    """ 
    return true number of action steps
    (dict, string, int, string, int) -> list, list, int
    """
    o_edgeID = canal_map[o_seg][1]
    d_edgeID = canal_map[d_seg][1]
    route = traci.simulation.findRoute(o_edgeID, d_edgeID)
    covered_segs = []
    for edgeID in route.edges:
        segments = edge_map[edgeID]
        if edgeID == o_edgeID:
            covered_segs.extend([s[2] for s in segments[o_seg - segments[0][2]:]])
        elif edgeID == d_edgeID:
            covered_segs.extend([s[2] for s in segments[0:d_seg - segments[0][2]]])
        else:
            covered_segs.extend([s[2] for s in segments])
    
    return route.edges, covered_segs, len(covered_segs)
    
def get_sim_step():
    """ 
    return current simulation time step in SUMO 
    """
    return int(traci.simulation.getTime())