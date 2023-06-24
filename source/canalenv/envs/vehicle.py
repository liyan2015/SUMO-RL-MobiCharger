# -*- coding: utf-8 -*-
"""
@file    vehicle.py
@author  Li Yan
@date    2020-05-26
@version 

Defines vehicles and their properties.
"""

# import traci
import libsumo as traci
import sumolib

class Vehicle:
    """ a vehicle """
    
    def __init__(
        self, 
        name: str = None,
        net: sumolib.net.Net = None,
        edge_map: dict = None,
        canal_map: dict = None,
        depart_step: int = 0,
        orig_speed: float = 0,
        MAX_STEP: int = 0
    ):
        self.KM = 1000
        self.FULLCHARGE_TIME = 10 #120 # sec
        self.CHARGER_DRIVING_RANGE = 0.5 * self.KM # 0.4 * self.KM # 0.3 * self.KM # 1 * self.KM # 
        
        self.SOC_THR = 0.1
        self.name = name
        self.SOC = 1.0
        self.SOC_cost = 0
        self.dist_cost = 0
        self.charge_amount = 1
        self.current_route = traci.vehicle.getRoute(self.name)
        self.edge_map = edge_map
        self.canal_map = canal_map
        if "charger" in self.name:
            self.orig_speed = orig_speed
            self.current_edgeID = self.current_route[0]
            self.current_pos = 0
            self.consume_rate = self.SOC / self.CHARGER_DRIVING_RANGE
            self.target_prev_dist = MAX_STEP * orig_speed
            self.target_cur_dist = self.target_prev_dist
            self.target_vehID = None
            self.target_charge_station = None
        else:
            self.orig_speed = orig_speed
            (
                self.self_canal_map, 
                self.self_edge_map, 
                self.self_route
            ) = self.self_canal_map_gen(net, net.getEdges(), self.orig_speed, self.current_route)
            self.cur_route_index = 0
            self.current_edgeID = self.self_canal_map[self.self_route[self.cur_route_index]][1]
            self.current_pos = self.self_canal_map[self.self_route[self.cur_route_index]][2]
            self.consume_rate = self.SOC / sum([net.getEdge(e).getLength() for e in self.current_route])
        self.last_edgeID = self.current_edgeID
        self.last_pos = self.current_pos
        self.exhaust_edgeID = self.current_edgeID
        self.exhaust_pos = self.current_pos
        self.move_to(net, self.current_edgeID, self.current_pos, lane=0)
        segments = self.edge_map[self.current_edgeID]
        self.current_seg = None
        for s in segments:
            if self.current_pos >= s[0] and self.current_pos < s[1]:
                self.current_seg = s[2]
        self.last_seg = self.current_seg
        self.previous_route = []
        self.total_distance = 0.0
        self.total_charge_time = 0.0
        self.total_charged_SOC = 0
        self.SOC_records = []
        self.charge_rate = 1 / float(self.FULLCHARGE_TIME)
        self.depart_step = depart_step
        self.exhaust_step = -1
        self.arrive_step = 0
        self.charge_dist = 0
        self.stopped = False
        self.be_charged = False
        self.stay_time = 0
        self.charging_others = 0
        self.charge_self = 0
        self.charge_others_count = 0
        self.last_charging_others = self.charging_others
        self.charge_others_seg = None
        self.neighbor_vehIDs = []
        self.step_charged_SOC = 0.0
        self.exhausted = False
        
        ## vessel metrics
        self.be_charge_count = 0
        self.total_delay = 0
        self.charge_delay = 0
        self.wait_charge_delay = 0
        self.wait_charge_delay_records = []
        self.SOC_samples = {}
        
        ## charger metrics
        self.charged_vessels = {}
        self.charge_amount = 0
        self.dist_log = []
        
        traci.vehicle.setSpeed(self.name, 0.001)
    
    def self_canal_map_gen(
        self,
        net: sumolib.net.Net = None,
        all_edges: list = None,
        speed: float = 0.0,
        route: list = None
    ):
        """ 
        partition vehicle route into segments by vehicle speed 
        """
        canal_map = {}
        tmp_canal_map = {}
        seg_index = 0
        edge_map = {}
        for edgeID in route:
            e = net.getEdge(edgeID)
            edge_len = e.getLength()
            seg_count = int(edge_len / speed)
            segments = []
            if seg_count > 0:
                for i in range(seg_count):
                    if i == seg_count - 1:
                        branches = [ed.getID() for ed in list(e.getOutgoing().keys()) if ed.getID() in route]
                        segments.append((i*speed, edge_len, seg_index, branches))
                        tmp_canal_map[seg_index] = (True, branches, edgeID, i*speed, edge_len)
                    else:
                        segments.append((i*speed, (i+1)*speed, seg_index, seg_index + 1))
                        tmp_canal_map[seg_index] = (False, seg_index + 1, edgeID, i*speed, (i+1)*speed)
                    
                    seg_index += 1
            else:
                branches = [ed.getID() for ed in list(e.getOutgoing().keys()) if ed.getID() in route]
                segments.append((0, edge_len, seg_index, branches))
                tmp_canal_map[seg_index] = (True, branches, edgeID, 0, edge_len)
                seg_index += 1
                
            edge_map[edgeID] = segments
            
        for s in range(seg_index):
            segment = tmp_canal_map[s]
            branches = [s]
            if segment[0]:
                for eID in segment[1]:
                    branches.append(edge_map[eID][0][2])
            else:
                branches.append(segment[1])
            canal_map[s] = (branches, segment[2], segment[3], segment[4])
        
        canal_route = []
        for edgeID in route:
            canal_route.extend([e[2] for e in edge_map[edgeID]])
        
        return canal_map, edge_map, canal_route
    
    def change_target(self, edgeID):
        """ 
        change current route to aim for a new target 
        """
        traci.vehicle.changeTarget(self.name, edgeID)
        self.current_route = traci.vehicle.getRoute(self.name)

    def get_cur_edgeID(self):
        """ 
        return current edgeID 
        """
        if "charger" in self.name:
            return self.canal_map[self.current_seg][1]
        else:
            if self.cur_route_index < len(self.self_route):
                return self.self_canal_map[self.self_route[self.cur_route_index]][1]
            else:
                return self.self_canal_map[self.self_route[-1]][1]
        
    def get_cur_pos(self):     
        """ 
        return current position 
        """
        if "charger" in self.name:
            return self.canal_map[self.current_seg][2]
        else:
            if self.cur_route_index < len(self.self_route):
                return self.self_canal_map[self.self_route[self.cur_route_index]][2]
            else:
                return self.self_canal_map[self.self_route[-1]][3]
        
    def get_cur_seg(self):
        self.current_edgeID = self.get_cur_edgeID()
        self.current_pos = self.get_cur_pos()
        segments = self.edge_map[self.current_edgeID]
        for s in segments:
            if self.current_pos >= s[0] and self.current_pos < s[1]:
                self.current_seg = s[2]
        return self.current_seg
        
    def get_cur_route(self):
        return traci.vehicle.getRoute(self.name)
        
    def get_cur_route_index(self):
        return traci.vehicle.getRouteIndex(self.name)
        
    def get_cur_speed(self):     
        """ 
        return current speed 
        """
        return traci.vehicle.getSpeed(self.name)
        
    def set_speed(self, speed):
        traci.vehicle.setSpeed(self.name, speed)
        
    def get_future_pos(self, est_future_pos, net):     
        """ 
        return future position after est_future_pos
        """
        current_route = traci.vehicle.getRoute(self.name)
        speed = traci.vehicle.getSpeed(self.name)
        current_edgeID = traci.vehicle.getRoadID(self.name)
        current_pos =  traci.vehicle.getLanePosition(self.name)
        est_travel_dist = speed * est_future_pos
        est_edgeID = current_edgeID
        est_pos = 0.0
        tmp_dist = 0.0

        for edgeID in current_route[traci.vehicle.getRouteIndex(self.name):]:
            edge = net.getEdge(edgeID)
            if edgeID == current_edgeID:
                tmp_dist += edge.getLength() - current_pos
            else:
                tmp_dist += edge.getLength()
            
            if tmp_dist > est_travel_dist:
                est_edgeID = edgeID
                est_pos = edge.getLength() - (tmp_dist - est_travel_dist)
                break

        if tmp_dist < est_travel_dist:
            est_edgeID = "arrived"
            est_pos = -1
        return est_edgeID
        
    def has_stop(self):
        return traci.vehicle.getNextStops(self.name)
        
    def is_stopped(self):
        return traci.vehicle.isStopped(self.name)
        
    def resume(self):
        return traci.vehicle.resume(self.name)
        
    def get_dist(self):
        return traci.vehicle.getDistance(self.name)
    
    def move_to(self, net, edgeID, pos, lane=0):
        traci.vehicle.moveTo(self.name, net.getEdge(edgeID).getLane(lane).getID(), pos)
    
    def update_SOC(self, net, sim_step_count, verbose = False):
        """ 
        update vehicle movement by route or charge event, and change of SOC 
        """
        if sim_step_count != self.depart_step and self.cur_route_index < len(self.self_route):
            current_self_seg = self.self_route[self.cur_route_index]
            self.current_edgeID = self.get_cur_edgeID()
            self.current_pos = self.get_cur_pos()
            
            if (self.SOC < self.SOC_THR and not self.stopped) or self.be_charged:
                self.exhaust_edgeID = self.current_edgeID
                self.exhaust_pos = self.current_pos
                self.stopped = True
                if not self.be_charged:
                    self.exhausted = True
                    self.exhaust_step = sim_step_count
            elif self.SOC > self.SOC_THR:
                self.total_distance += self.self_canal_map[current_self_seg][3] - self.self_canal_map[current_self_seg][2]
                self.SOC -= self.consume_rate * (self.self_canal_map[current_self_seg][3] - self.self_canal_map[current_self_seg][2])
                self.cur_route_index += 1
                if self.cur_route_index < len(self.self_route):
                    current_self_seg = self.self_route[self.cur_route_index]
                    next_edgeID = self.self_canal_map[current_self_seg][1]
                    next_pos = self.self_canal_map[current_self_seg][2]
                else:
                    next_edgeID = self.self_canal_map[current_self_seg][1]
                    next_pos = self.self_canal_map[current_self_seg][3]
                self.move_to(net, next_edgeID, next_pos)
            
            if self.stopped and not self.be_charged and self.SOC > self.SOC_THR:
                self.charge_dist = self.total_distance
                self.stopped = False
            elif self.stopped:
                self.move_to(net, self.exhaust_edgeID, self.exhaust_pos)
                if self.SOC < self.SOC_THR:
                    self.wait_charge_delay += 1

            if self.be_charged:                  
                self.SOC += 1 * self.charge_rate
                self.total_charged_SOC += 1 * self.charge_rate
                self.step_charged_SOC = 1 * self.charge_rate
                self.charge_dist = self.total_distance
                self.stopped = False
                self.charge_delay += 1
                self.be_charge_count += 1
                if self.SOC >= self.SOC_THR:
                    self.exhausted = False
                    if self.wait_charge_delay > 0:
                        self.wait_charge_delay_records.append(self.wait_charge_delay)
                        self.wait_charge_delay = 0
                        
        elif self.cur_route_index == len(self.self_route):
            self.arrive_step = sim_step_count
            traci.vehicle.remove(self.name)
    
    def update(self, net, action, orig_action, chg_canalIDs, veh_collection, can_charge_others, verbose=False):
        """ 
        update charger movement by action, and change of SOC 
        """       
        # determine current position !!! traci.vehicle.getRoadID is not reliable, may return internal edgeID
        self.current_edgeID = self.get_cur_edgeID()
        self.current_pos = self.get_cur_pos()
        self.current_seg = action    
        self.dist_cost = 0
        
        target_seg = self.canal_map[action]
        if self.current_edgeID != target_seg[1]:
            self.change_target(target_seg[1])
        
        ## orig_action == 2~4
        if self.current_seg != self.last_seg:
            self.dist_cost = self.canal_map[self.last_seg][3] - self.canal_map[self.last_seg][2]
            self.total_distance += self.dist_cost
            self.SOC -= self.consume_rate
            if self.charge_others_seg != None and self.last_seg == self.charge_others_seg:
                self.SOC_cost = 0
                self.charge_others_seg = None
            self.SOC_cost += self.dist_cost
            self.stay_time = 0
            self.charging_others = 0
            # self.charge_self = 0
            
        ## orig_action == 0 or 1
        elif self.current_seg == self.last_seg: 
            ## stays to charge others
            if orig_action == 1:
                self.charging_others = 1
                self.charge_others_seg = self.current_seg
                if self.last_charging_others != self.charging_others:
                    self.stay_time = 0
                self.stay_time += 1
            else:
                ## stays at a charger, SoC refilled
                if (
                    self.current_seg in chg_canalIDs and
                    self.SOC < 1 
                    ):
                    self.charge_dist = self.total_distance
                    self.charge_self = 1

                    if self.SOC + 1 * self.charge_rate < 1:
                        self.SOC += 1 * self.charge_rate
                        self.charge_amount += 1 * self.charge_rate
                        self.step_charged_SOC = 1 * self.charge_rate
                    else:
                        self.charge_amount += 1 - self.SOC
                        self.step_charged_SOC = 1 - self.SOC
                        self.SOC = 1
                # else:
                    # self.charge_self = 0
                    
                self.charging_others = 0
                if self.last_charging_others != self.charging_others:
                    self.stay_time = 0
                    
                ## stays at a canal segment
                self.stay_time += 1
        
        self.move_to(net, target_seg[1], target_seg[2])
        
        # store last known edgeID
        self.last_edgeID = self.current_edgeID
        self.last_pos = self.current_pos
        self.last_seg = self.current_seg
        self.last_charging_others = self.charging_others
        
        return self.neighbor_vehIDs
        
    def get_output(self):
        """ 
        print simulation metrics of vessels and chargers 
        """
        if "charger" not in self.name:
            wait_charge_delay_records = ','.join(str(e) for e in self.wait_charge_delay_records)
            return [self.name, self.total_delay, self.charge_delay, 
                wait_charge_delay_records, self.be_charge_count,
                self.depart_step, self.arrive_step, 
                self.total_charged_SOC, len(self.self_route),
                self.SOC_samples]
        else:
            return [self.name, self.total_distance, self.charge_amount, self.charged_vessels, self.dist_log] 