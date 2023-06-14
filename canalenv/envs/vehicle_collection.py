# -*- coding: utf-8 -*-
"""
@file    vehicleCollection.py
@author  Li Yan
@date    2020-05-26
@version 

Defines a collection of vehicles.
"""

# import traci
import libsumo as traci
import os, csv

class VehicleCollection:
    """ collection of vehicles """

    def __init__(self):
        self.fleet = {}

    def add_vehicle(self, vehicle):
        """ 
        add a vehicle to the collection 
        (Vehicle)
        """
        self.fleet[vehicle.name] = vehicle

    def stop_vehicle(self, vehicleID, step):
        """ 
        list a vehicle as stopped once it has left the simulation
        (string, int)
        """
        if vehicleID in self.fleet:
            self.fleet[vehicleID].stop(step)

    def get_values(self):
        """ 
        return a list of vehicles
        () -> Vehicle[]
        """
        return self.fleet.values()
    
    def has_vehicle(self, vehicleID):
        """ 
        return whether has the vehicle
        (string) -> Boolean
        """
        return vehicleID in self.fleet

    def get_vehicle(self, vehicleID):
        """ 
        return a particular vehicle 
        (string) -> Vehicle 
        """
        return self.fleet[vehicleID]

    def remove_vehicles(self, step):
        traci.switch(traci.getLabel())
        for v in self.fleet:
            if self.fleet[v].parked == 900 or self.fleet[v].end == step:
                traci.vehicle.remove(v)
                
    def output(
        self, 
        folder: str = "./", 
        label: str = "default", 
        code: str ="vessel", 
        SAMPLE_INTERVAL: int = 1800, 
        total_steps: int = 3600
    ):
        """ 
        print simulation metrics of all vehicles, to vehicle.out.csv in an optionally-specified folder 
        """
        DAY_SECS = 24 * 3600
        if code == "vessel":
            result_path = os.path.join(folder, label + '-vessel.out.csv')
            with open(result_path, 'w', newline='') as csvfile:
                vehicle_writer = csv.writer(csvfile, delimiter=',')
                vehicle_writer.writerow(["vesselID","total_delay",
                                        "charge_delay", "wait_charge_delay_records", 
                                        "be_charge_count", "depart_step", "arrive_step",
                                        "total_charged_SOC", "route_length", 
                                        "SOC_samples"])
                for vehicleID, vehicle in self.fleet.items():
                    if "charger" not in vehicleID: 
                        veh_output = vehicle.get_output()
                        full_SOC_samples = []
                        for i in range(int(total_steps / SAMPLE_INTERVAL)):
                            if i * SAMPLE_INTERVAL in veh_output[-1]:
                                full_SOC_samples.append(veh_output[-1][i * SAMPLE_INTERVAL])
                            else:
                                full_SOC_samples.append(-1)
                        veh_output[-1] = full_SOC_samples
                        vehicle_writer.writerow(veh_output)
        elif code == "charger":
            result_path = os.path.join(folder, label + '-charger.out.csv')
            with open(result_path, 'w', newline='') as csvfile:
                vehicle_writer = csv.writer(csvfile, delimiter=',')
                vehicle_writer.writerow(["chargerID","total_distance",
                                        "charge_amount", "charged_vessels",
                                        "dist_log"])
                for vehicleID, vehicle in self.fleet.items():
                    if "charger" in vehicleID: 
                        veh_output = vehicle.get_output()
                        charged_vessel_count = [0] * (int(total_steps / SAMPLE_INTERVAL) + 1)
                        for sample_interval, vehicleIDs in veh_output[3].items():
                            charged_vessel_count[sample_interval] += len(vehicleIDs)
                        veh_output[3] = ','.join(str(e) for e in charged_vessel_count)
                        veh_output[4] = ','.join(str(e) for e in veh_output[4])
                        vehicle_writer.writerow(veh_output)