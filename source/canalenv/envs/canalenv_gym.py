# -*- coding: utf-8 -*-
"""
@file    canalenv_gym.py
@author  Li Yan
@date    2020-05-26
@version 

SUMO Reinforcement Learning environment for Dispatching of Mobile Chargers.
"""

import os, sys, sumolib
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
# import traci
import libsumo as traci
import numpy as np
import gym
from gym import spaces
import canalenv
from canalenv.envs.vehicle import Vehicle
from canalenv.envs.vehicle_collection import VehicleCollection
from canalenv.envs.utils import *

class SumoEnv(gym.Env):
    """ SUMO Reinforcement Learning environment for Dispatching of Mobile Chargers.
    A gym.Env interface for dispatching of mobile chargers using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.
    Args:
        label (str): can be "train", "test" or "evaluate", used for differentiating environment instances, configuration files and route files when training in parallel.
        gui_f (bool): whether to run SUMO simulation with the SUMO GUI.
        env_id (str): 'SumoEnv-v0'.
        num_charger (int): number of chargers for dispatching.
    """

    def __init__(
        self, 
        label: str, 
        gui_f: bool = False, 
        env_id: str = 'SumoEnv-v0', 
        num_charger: int = 4,
        date = '8_9' # '8_7' # '8_8' # '8_6' # 
    ):
        """ Initialize the environment. """
        super(SumoEnv, self).__init__()
        self.MAX_STEP = 300000 # 250000 # 500000 # 400000 # 80000 # 30000 # 
        self.MAX_SPEED = 10
        self.SPEED_TIMES = 15 # 10 # 5 # 1 # 20 # 50 # 30 # 
        self.INF_DIST = 1e30
        self.SAMPLE_INTERVAL = 3600 / 2
        self.KM = 1000
        self.CHARGER_DRIVING_RANGE = 0.5 * self.KM # 1 * self.KM # 0.4 * self.KM # 0.3 * self.KM # 
        
        self.label = label
        self.env_id = env_id      
        os.chdir(os.path.dirname(__file__))
        self.work_dir = os.getcwd()
        sumo_path = os.path.join(self.work_dir, 'Input')
        self.sumo_config = os.path.join(sumo_path, self.label + "_canal.sumocfg")
        net_file = "canal_3lane.net_uturn.xml"
        net_path = os.path.join(sumo_path, net_file)
        route_map = {
            '8_6':'86.1hrs.rou.xml', # "86.24hrs.rou.xml", #
            '8_7':'87.1hrs.rou.xml', # '87.24hrs.rou.xml', #
            '8_8':'88.1hrs.rou.xml', # '88.24hrs.rou.xml', #
            '8_9':'89.1hrs.rou.xml', # '89.24hrs.rou.xml', #
        }
        route_file = route_map[date]
        route_path = os.path.join(sumo_path, route_file)
        charger_route_file = self.label + "_charger.rou.xml"
        charger_route_path = os.path.join(sumo_path, charger_route_file)
        charger_path = sumo_path
        self.num_charger = num_charger
        self.exe = 'sumo-gui' if gui_f else 'sumo'
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', self.exe)
        self.sumo_cmd = [sumo_binary, '-c', self.sumo_config]
        self.env_exist = False
        
        ## generate sumo configuration file
        # if not os.path.exists(self.sumo_config):
        create_sumo_cfg(
            self.sumo_config, 
            net_file, 
            route_file, 
            charger_route_file, 
            self.label
        )

        ## use sumolib to obtain the network info and edge info
        (
            self.net, 
            all_edges, 
            self.chargerIDs, 
            start_edgeIDs
        ) = prep_canal_net(net_path, num_charger)
        
        ## partition edges into segments by speed limit
        (
            self.canal_map, 
            self.edge_map
        ) = canal_map_gen(
            all_edges, 
            self.MAX_SPEED, 
            self.SPEED_TIMES
        )
        
        ## generate self.label_charger.rou.xml file for chargers
        self.charger_canalIDs = prep_veh_route(
            self.net, 
            self.edge_map, 
            charger_route_path, 
            self.num_charger, 
            0, 
            start_edgeIDs, 
            charger_path
        )
        
        ## read vessel and charger info from their route XML file
        (
            self.veh_depart_time,
            self.veh_info,
            self.veh_orig_speed,
            self.max_veh_num
        ) = prep_veh_depart(route_path, charger_route_path, self.MAX_SPEED, self.SPEED_TIMES)
        
        ## 0:'stay', 1:'charge at', 2~5:'go to'
        self.max_act_num = max([len(list(e.getOutgoing().keys())) for e in all_edges]) + 2
        state_len = len(self.canal_map) + self.num_charger * 7 + 2 * self.num_charger * self.max_act_num + self.num_charger * len(self.charger_canalIDs)
        self.action_space = spaces.MultiDiscrete([self.max_act_num] * self.num_charger)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_len,), dtype=np.float32)
        self.veh_collection = VehicleCollection()
        self._state = -np.ones((state_len,), dtype=np.float32)
        self._episode_ended = False
        self.step_count = 0
        self.sim_step_count = 0
        self.charged_SOC = 0
        self.charge_complete = False
        self.SOC_samples = {}
    
    def action_spec(self):
      return self._action_spec

    def observation_spec(self):
      return self._observation_spec
    
    def get_state_d(self, veh_list):      
        SOC_state = np.zeros((len(self.canal_map),), dtype=np.float32)
        charger_state = np.zeros((self.num_charger, 7), dtype=np.float32)
        charge_station_state = np.zeros((self.num_charger, len(self.charger_canalIDs)), dtype=np.float32)
        elig_act_state = np.zeros((self.num_charger, self.max_act_num), dtype=np.float32)
        dir_state = np.zeros((self.num_charger, self.max_act_num), dtype=np.float32)
        
        chr_cur_seg_pos = [
            (self.veh_collection.get_vehicle(c).get_cur_edgeID(), 
            self.veh_collection.get_vehicle(c).get_cur_pos(), 
            self.veh_collection.get_vehicle(c).get_cur_seg()) for c in self.chargerIDs
        ]
        
        for c in self.chargerIDs:
            charger = self.veh_collection.get_vehicle(c)
            charger.neighbor_vehIDs = []
        
        checked_vehIDs = []
        for vehicleID in veh_list:
            if "charger" not in vehicleID:
                vessel = self.veh_collection.get_vehicle(vehicleID)
                cur_segID = vessel.get_cur_seg()
                if (
                    vessel.total_charged_SOC < vessel.SOC_THR 
                    and vessel.SOC <= 1 - vessel.SOC_THR
                   ):
                    SOC_state[cur_segID] += 1 / self.max_veh_num
                    for charger_index, c in enumerate(self.chargerIDs):
                        if (
                            vehicleID not in checked_vehIDs
                            and chr_cur_seg_pos[charger_index][2] == cur_segID 
                            ):
                            charger = self.veh_collection.get_vehicle(c)
                            charger.neighbor_vehIDs.append(vehicleID)
                            checked_vehIDs.append(vehicleID)

        for charger_index, c in enumerate(self.chargerIDs):
            charger = self.veh_collection.get_vehicle(c)
            chr_cur_seg = chr_cur_seg_pos[charger_index][2]
            charger_state[charger_index, 0] = chr_cur_seg / len(self.canal_map)
            charger_state[charger_index, 1] = charger.stay_time / self.MAX_STEP
            charger_state[charger_index, 2] = charger.charging_others
            charger_state[charger_index, 3] = charger.charge_self
            charger_state[charger_index, 4] = charger.SOC
            nearest_chg_stn = None
            max_remain_SOC = 0
            for chg_stn_index, chg_stn in enumerate(self.charger_canalIDs):
                d_edgeID = self.canal_map[chg_stn][1]
                d_pos = self.canal_map[chg_stn][2]
                (_, _, route_num_steps) = get_route(self.edge_map, self.canal_map, chr_cur_seg, chg_stn)
                arrive_SOC = charger.SOC - route_num_steps * charger.consume_rate
                if arrive_SOC > max_remain_SOC:
                    nearest_chg_stn = chg_stn
                    max_remain_SOC = arrive_SOC
                charge_station_state[charger_index, chg_stn_index] = arrive_SOC / (charger.consume_rate * self.MAX_STEP)
            
            if charger.target_vehID != None:
                vessel = self.veh_collection.get_vehicle(charger.target_vehID)
                cur_segID = vessel.get_cur_seg()
                charger.target_prev_dist = get_distance(self.canal_map, chr_cur_seg, cur_segID)
            else:
                chr_target_vehIDs = [self.veh_collection.get_vehicle(c).target_vehID for c in self.chargerIDs]
                charger.target_prev_dist = self.MAX_SPEED * self.SPEED_TIMES * self.MAX_STEP
                for vehicleID in veh_list:
                    if "charger" not in vehicleID:
                        vessel = self.veh_collection.get_vehicle(vehicleID)
                        if (
                            vehicleID not in chr_target_vehIDs
                            and vessel.total_charged_SOC < vessel.SOC_THR 
                            and vessel.SOC <= 1 - vessel.SOC_THR
                            ):
                            cur_segID = vessel.get_cur_seg()
                            min_other_dist = self.MAX_SPEED * self.SPEED_TIMES * self.MAX_STEP
                            for other_index, tmp_target_vehID in enumerate(chr_target_vehIDs):
                                if other_index != charger_index and tmp_target_vehID == None:
                                    other_seg = chr_cur_seg_pos[other_index][2]
                                    tmp_other_drive_dist= get_distance(self.canal_map, other_seg, cur_segID)
                                    if tmp_other_drive_dist < min_other_dist:
                                        min_other_dist = tmp_other_drive_dist
                                    
                            tmp_drive_dist= get_distance(self.canal_map, chr_cur_seg, cur_segID)
                            if tmp_drive_dist < charger.target_prev_dist and tmp_drive_dist < min_other_dist:
                                charger.target_prev_dist = tmp_drive_dist
                                charger.target_vehID = vehicleID
            
            charger_state[charger_index, 5] = charger.target_prev_dist / (self.MAX_SPEED * self.SPEED_TIMES * self.MAX_STEP)
                
            ## 1 - charger c can potentially charge others; 0 - charger c has no nearby vessels
            charger_state[charger_index, 6] = 1 if charger.neighbor_vehIDs else 0
            
            tmp_cand_segs = self.canal_map[chr_cur_seg][0]
            tmp_cand_indices = list(range(len(tmp_cand_segs)))
            elig_act_state[charger_index, np.array(tmp_cand_indices)] = 1
            
            ## Determine the best action at current state
            if charger.target_vehID != None:
                vessel = self.veh_collection.get_vehicle(charger.target_vehID)
                d_seg = vessel.get_cur_seg()
                    
                min_dist = self.MAX_SPEED * self.SPEED_TIMES * self.MAX_STEP
                min_index = 0
                for i, cand_seg in enumerate(tmp_cand_segs):
                    tmp_drive_dist= get_distance(self.canal_map, cand_seg, d_seg)
                    if tmp_drive_dist < charger.target_prev_dist and tmp_drive_dist < min_dist:
                        min_dist = tmp_drive_dist
                        min_index = i
                        
                ## When there are vessels nearby, the best action should be charging them
                if charger.neighbor_vehIDs:
                    min_index = 1
            else:
                ## When there are no vessels running out of SOC, the best action should be stay
                min_index = 0
            dir_state[charger_index, min_index] = 1
            
            if chr_cur_seg in self.charger_canalIDs and charger.SOC < 1 :
                dir_state[charger_index, 0] = 1
        
        return np.concatenate([
            SOC_state, 
            charger_state.reshape(self.num_charger * 7,),
            elig_act_state.reshape(self.num_charger * self.max_act_num,),
            dir_state.reshape(self.num_charger * self.max_act_num,),
            charge_station_state.reshape(self.num_charger * len(self.charger_canalIDs),),
        ])

    def step(self, action):
        ## 0 - stay, 1 - charge, 2 - 4 go possible downstream segments
        self.step_count += 1
        reward = 0.0
        
        ## current road segments of all chargers
        all_chr_cur_segs = [self.veh_collection.get_vehicle(c).get_cur_seg() for c in self.chargerIDs]
        
        ## extract the eligible actions of each charger given its current road segment
        cand_segs = {}
        cand_indices = {}
        for c in range(self.num_charger):
            tmp_cand_segs = self.canal_map[all_chr_cur_segs[c]][0]
            tmp_cand_indices = list(range(len(tmp_cand_segs)))
            cand_segs[c] = tmp_cand_segs
            cand_indices[c] = tmp_cand_indices
        
        orig_action = action
        elig_action = [action[a] in cand_indices[a] for a in range(self.num_charger)]
        all_has_SOC = True
        all_charged = []
        if all(elig_action):
            action = [cand_segs[c][action[c]] for c in range(self.num_charger)]
            traci.simulationStep()
            veh_list = traci.vehicle.getIDList()
            self.sim_step_count = get_sim_step()
            
            if self.sim_step_count in self.veh_depart_time:
                for vehicleID in self.veh_depart_time[self.sim_step_count]:
                    depart_delayed = False
                    if vehicleID not in traci.vehicle.getLoadedIDList():
                        ## Make sure all vehicles depart as scheduled
                        traci.vehicle.add(vehicleID, 
                            "",
                            typeID=self.veh_info[vehicleID]["type"], 
                            depart='now', 
                            departLane='free', 
                            departPos='last', 
                            departSpeed='max'
                        )
                        traci.vehicle.setRoute(vehicleID,self.veh_info[vehicleID]["routeEdges"])
                        traci.vehicle.moveTo(vehicleID, self.net.getEdge(self.veh_info[vehicleID]["routeEdges"][0]).getLane(0).getID(), 0)
                        depart_delayed = True
                        veh_list = traci.vehicle.getIDList()
                    
                    self.veh_collection.add_vehicle(Vehicle(
                        vehicleID, self.net, self.edge_map, 
                        self.canal_map, self.sim_step_count, 
                        self.veh_orig_speed[traci.vehicle.getTypeID(vehicleID)],
                        self.MAX_STEP,
                        self.CHARGER_DRIVING_RANGE
                        )
                    )
                    if depart_delayed:
                        ## https://stackoverflow.com/questions/66582896/sumo-traci-cannot-add-a-vehicle-after-removing-it-without-calling-simulationst
                        ## Vehicles which leave the simulation in the current simulation step are only 
                        ## removed at the end of the step while traci commands are handled in the beginning.
                        traci.vehicle.remove(vehicleID)
            
            for vehicleID in veh_list:
                if "charger" not in vehicleID:
                    vessel = self.veh_collection.get_vehicle(vehicleID)
                    vessel.update_SOC(self.net, self.sim_step_count, verbose=False)
                
            for vehicleID in veh_list:
                if "charger" in vehicleID:
                    reward_given = False
                    tmp_reward = 0.0
                    charger = self.veh_collection.get_vehicle(vehicleID)
                    charger_index = self.chargerIDs.index(vehicleID)
                    can_charge_others = self._state[len(self.canal_map) + charger_index * 7 + 6]
                    before_SOC = charger.SOC
                    charged_vehIDs = charger.update(
                        self.net,
                        action[charger_index], 
                        orig_action[charger_index],
                        self.charger_canalIDs,
                        self.veh_collection,
                        can_charge_others
                    )
                    
                    if charger.charging_others == 1 and charged_vehIDs:
                        min_SOC = self.veh_collection.get_vehicle(charged_vehIDs[0]).SOC
                        target_veh = charged_vehIDs[0]
                        for charged_vehID in charged_vehIDs:
                            if self.veh_collection.get_vehicle(charged_vehID).SOC < min_SOC:
                                min_SOC = self.veh_collection.get_vehicle(charged_vehID).SOC
                                target_veh = charged_vehID
                                
                        ## charge a vessel
                        charger.charge_others_count += 1
                        target_veh = self.veh_collection.get_vehicle(target_veh)
                        target_veh.be_charged = True
                        
                        ## demo charging process
                        if self.label == "test" and self.exe == 'sumo-gui':
                            charge_edgeID = target_veh.get_cur_edgeID()
                            charge_pos = target_veh.get_cur_pos()
                            charge_pos = charge_pos - 5 if charge_pos - 5 > self.canal_map[action[charger_index]][2] else charge_pos + 10
                            charger.move_to(self.net, charge_edgeID, charge_pos)
                            traci.vehicle.highlight(target_veh.name, (0, 255, 0, 255), size=5, alphaMax=-1, duration=1)
                        
                        target_veh.update_SOC(self.net, self.sim_step_count, verbose=False)
                        step_charged_SOC = target_veh.step_charged_SOC
                        self.charged_SOC += step_charged_SOC
                        
                        ## reward charger for charging a vessel
                        tmp_reward += 2
                        
                        if target_veh.name == charger.target_vehID:
                            charger.target_vehID = None
                            
                        chr_target_vehIDs = [self.veh_collection.get_vehicle(c).target_vehID for c in self.chargerIDs]
                        if target_veh.name in chr_target_vehIDs:
                            self.veh_collection.get_vehicle(self.chargerIDs[chr_target_vehIDs.index(target_veh.name)]).target_vehID = None
                            
                        if self.label == "test":
                            sample_interval = int(self.sim_step_count / self.SAMPLE_INTERVAL)
                            if sample_interval in charger.charged_vessels:
                                if target_veh.name not in charger.charged_vessels[sample_interval]:
                                    charger.charged_vessels[sample_interval].append(target_veh.name)
                            else:
                                charger.charged_vessels[sample_interval] = [target_veh.name]
                            
                        reward_given = True
                    
                    ## + reward charger for SOC refill
                    if charger.charge_self == 1:
                        tmp_reward += 3 * charger.step_charged_SOC + 0.5 * (1 - before_SOC)
                            
                        if self.label == "test" and self.exe == 'sumo-gui':
                            traci.vehicle.highlight(charger.name, (0, 0, 255, 255), size=5, alphaMax=-1, duration=1)
                            
                        reward_given = True
                    
                    if not reward_given:
                        dir_start = len(self.canal_map) + self.num_charger * 7 + self.num_charger * self.max_act_num + charger_index * self.max_act_num
                        dir_end = len(self.canal_map) + self.num_charger * 7 + self.num_charger * self.max_act_num + (charger_index + 1) * self.max_act_num
                        dir_state = self._state[dir_start:dir_end]
                        if dir_state[orig_action[charger_index]] == 1:
                            tmp_reward += 8e-2
                        else:
                            tmp_reward += -8e-2
                            
                    reward += tmp_reward
                    
                    if self.label == "test":
                        print(vehicleID, tmp_reward, orig_action[charger_index], charger.target_vehID, charger.neighbor_vehIDs)
                            
            self._state = self.get_state_d(veh_list)
            
            if self.sim_step_count % self.SAMPLE_INTERVAL == 0 and self.label == "test":
                for vehicleID in veh_list:
                    vehicle = self.veh_collection.get_vehicle(vehicleID)
                    if "charger" not in vehicleID:
                        vehicle.SOC_samples[self.sim_step_count] = vehicle.SOC
                    else:
                        vehicle.dist_log.append(vehicle.total_distance)
                        if self.sim_step_count in self.SOC_samples:
                            self.SOC_samples[self.sim_step_count].append(vehicle.SOC)
                        else:
                            self.SOC_samples[self.sim_step_count] = [charger.SOC]
        else:
            reward = -8e-1
        
        if self.label == "test":
            print("Final reward:", reward, "Target vehicles", [self.veh_collection.get_vehicle(c).target_vehID for c in self.chargerIDs], "Target CSs", [self.veh_collection.get_vehicle(c).target_charge_station for c in self.chargerIDs])
        
        for vessel in self.veh_collection.get_values():
            if "charger" not in vessel.name:
                vessel.be_charged = False
                all_charged.append(vessel.total_charged_SOC >= vessel.SOC_THR)    
            else:
                vessel.charge_self = 0
                charger_index = self.chargerIDs.index(vessel.name)
                chrg_stn_state_start = len(self.canal_map) + self.num_charger * 7 + 2 * self.num_charger * self.max_act_num + charger_index * len(self.charger_canalIDs)
                chrg_stn_state_end = len(self.canal_map) + self.num_charger * 7 + 2 * self.num_charger * self.max_act_num + (charger_index + 1) * len(self.charger_canalIDs)
                if max(self._state[chrg_stn_state_start:chrg_stn_state_end]) < 0:
                    reward = -300
                    vessel.exhausted = True
                    all_has_SOC = False
                
        self.charge_complete = bool(all_charged) and all(all_charged) and len(all_charged) == self.max_veh_num
        info = {}

        if (
            self.step_count >= self.MAX_STEP 
            or not all_has_SOC 
            or self.charge_complete
            ):
            if self.charge_complete:
                reward = 250
                if self.label == "test":
                    for vessel in self.veh_collection.get_values():
                        if "charger" not in vessel.name:
                            if vessel.arrive_step != 0:
                                vessel.total_delay = vessel.arrive_step - vessel.depart_step - len(vessel.self_route)
                            else:
                                vessel.total_delay = vessel.charge_delay + sum(vessel.wait_charge_delay_records)
                    self.output()
                
            self._episode_ended = True
            info["TimeLimit.truncated"] = self.step_count >= self.MAX_STEP #or not all_has_SOC
            info["charge_complete"] = self.charge_complete
            info["steps"] = self.step_count
            info["sim_step_count"] = self.sim_step_count
        
            for c in self.chargerIDs:
                charger = self.veh_collection.get_vehicle(c)
                if charger.exhausted:
                    info[c + " exhausted"] = True
                    
        
        return self._state, reward, self._episode_ended, info
    
    def reset(self):
        # ############ traci #################
        # if self.port != None:
            # sumo_cmd = ['-c', self.sumo_config]
            # traci.load(sumo_cmd)
        # else:
            # self.port = sumolib.miscutils.getFreeSocketPort()
            # traci.start(self.sumo_cmd, label=self.label, port=self.port, traceFile="/usr/data2/sumo/" + self.label + "_traci_commands.log")
        
        ############ libsumo #################
        if self.env_exist:
            traci.close()
        self.env_exist = True
        traci.start(self.sumo_cmd)
        
        self._episode_ended = False
        self.step_count = 0
        self.sim_step_count = get_sim_step()
        self.charged_SOC = 0
        self.charge_complete = False
        self.veh_collection = VehicleCollection()
        
        for vehicleID in self.veh_depart_time[self.sim_step_count]:
            self.veh_collection.add_vehicle(Vehicle(
                vehicleID, self.net, self.edge_map, 
                self.canal_map, self.sim_step_count, 
                self.veh_orig_speed[traci.vehicle.getTypeID(vehicleID)],
                self.MAX_STEP,
                self.CHARGER_DRIVING_RANGE
                )
            )
            
        # spawn the vehicles at time 0
        veh_list = traci.vehicle.getIDList()
        self._state = self.get_state_d(veh_list)
        return self._state

    def close(self):
        traci.close()
        
    def output(self):
        output_path = os.path.join(self.work_dir, 'Output')
        self.veh_collection.output(output_path, self.label, "vessel", self.SAMPLE_INTERVAL, self.sim_step_count)
        self.veh_collection.output(output_path, self.label, "charger", self.SAMPLE_INTERVAL, self.sim_step_count)