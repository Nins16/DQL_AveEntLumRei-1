import traci
from sumolib import checkBinary
import argparse
import sys
import os
import pandas as pd
from pathlib import Path

options = argparse.ArgumentParser()
options.add_argument("--gui", "-g", action="store_true")

args = options.parse_args()

GUI = args.gui

ENVIRONMENT_PATH = "Simulation_Environment\Main MADDPG"
SCHEDULER = Path(f"{ENVIRONMENT_PATH}\\traffic plans.csv")


class SumoEnvironment:
    def __init__(self, GUI, SUMO_CFG, scheduler):
        self.GUI = GUI
        self.SUMO_CFG = SUMO_CFG
        self.scheduler = scheduler
        
        if not self.GUI:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        traci.start([self.sumoBinary, "-c", f"{self.SUMO_CFG}\osm.sumocfg",
                     "--tripinfo-output", f"{self.SUMO_CFG}\\Results\\tripinfo.xml", "--start"])
        
        self.scheduler_maker(self.scheduler)

    def scheduler_maker(self,t_dir):
        tls_dict = {}

        #Get Critical Times from CSV
        df = pd.read_csv(t_dir)
        for i in range(df.shape[0]):
            p_id = df.iloc[i,0]
            times = df.iloc[i,1]
            tls_id = df.iloc[i,2]

            #Transforms int to list if int
            if type(times) == str:
                times = times.split('/')
            else:
                times = [times]
            #Use time as dct key
            for time in times:
                try:
                    tls_dict[time].append((p_id, tls_id))
                except KeyError:
                    tls_dict[time] = []
                    tls_dict[time].append((p_id, tls_id))
        
        self.schedule_switch_time = list(tls_dict.keys())
        self.tls_dict = tls_dict

    
    def switch(self, time):
        # print(self.tls_dict[0])
        time = str(time)
        if time in self.schedule_switch_time:
            critical = self.tls_dict[time]

            for p_id, tls in critical:
                traci.trafficlight.setProgram(tls, int(p_id))

    def step(self):
        self.switch(int(traci.simulation.getTime()))
        traci.simulation.step()

if __name__ == "__main__":
    env = SumoEnvironment(GUI,ENVIRONMENT_PATH, SCHEDULER)
    
    while traci.simulation.getMinExpectedNumber() != 0:
        env.step()






