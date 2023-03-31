import traci
from sumolib import checkBinary
import argparse
import sys
import os

options = argparse.ArgumentParser()
options.add_argument("--gui", "-g", action="store_true")

args = options.parse_args()

GUI = args.gui

ENVIRONMENT_PATH = "Simulation_Environment\Main MADDPG"
SCHEDULER = f"{ENVIRONMENT_PATH}\traffic_plans.xlsx"


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
        tls = pd.ExcelFile(t_dir).sheet_names
        tls_dict = {}

        for tl in tls:
            df = pd.read_excel(t_dir, sheet_name=tl)

            for i in range(df.shape[0]):
                p_id = df.iloc[i,0]
                begin_times = df.iloc[i,1]
                if type(begin_times) == str:
                    begin_times = begin_times.split('/')
                else:
                    begin_times = [begin_times]
                
                for time in begin_times:
                    try:
                        tls_dict[time].append((p_id, tl))
                    except KeyError:
                        tls_dict[time] = []
                        tls_dict[time].append((p_id, tl))
        self.schedule_switch_time = list(tls_dict.keys())
        self.tls_dict = tls_dict
    
    def switch(self, time):
        if int(time) in self.schedule_switch_time:
            critical = self.tls_dict[time]
            for p_id, tls in critical:
                traci.trafficlight.setProgram(tls, p_id)

    def step(self):
        self.switch(int(traci.simulation.getTime()))
        traci.simulation.step()

if __name__ == "__main__":
    env = SumoEnvironment(GUI,ENVIRONMENT_PATH, SCHEDULER)
    
    while traci.simulation.getMinExpectedNumber() != 0:
        
        env.step()






