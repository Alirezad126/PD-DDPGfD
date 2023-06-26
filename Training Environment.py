#Importing Libraries:
import math
import random
from typing import Optional
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

#Reading Data:
data = pd.read_csv("Training_data.csv")
all_data = pd.read_csv("All_answers.csv")
days_to_train = data["Day"].unique()
all_days = all_data["Day"].unique()

#maximum and minimum:
max_elec = max(all_data["Electricity"])
min_elec = min(all_data["Electricity"])
max_heat_load = max(all_data["Heating"])
min_heat_load = min(all_data["Heating"])
max_cold_load = max(all_data["Cooling"])
min_cold_load = min(all_data["Cooling"])
max_pv = max(all_data["PV Power"])
min_pv = min(all_data["PV Power"])
max_wind = max(all_data["Wind Power"])
min_wind = min(all_data["Wind Power"])
max_temp = max(all_data["Temperature"])
min_temp = min(all_data["Temperature"])

#Defining The CAES Object:
class CAES_Class(object):

    cp = 1.005
    P_amb = 1
    T_amb = 298
    P_min = 42
    P_max = 72
    time_step = 3600
    T_tank=298
    R = 287
    gamma = 1.4
        
    P_CAES_ch_max = 800
    P_CAES_dch_max = 800
    gamma = 1.4
    cp = 1.005
    T_amb = 298
    T_tank = 298
    V_tank = 1100.19
    
    def __init__(self,initial=57):
        self.pressure = initial
        self.SOC = round((self.pressure-self.P_min)/(self.P_max - self.P_min),ndigits=3)
        
    def discharging_mfr(self,P_discharge):
        term2 = 0.00278671 * P_discharge + 1.2292957997285587

        return term2


    def cooling_load_predicted(self,P_discharge,t_amb):
        term1 = 0.39792147*P_discharge + 432.3243866973607
        term2 = 0.00278671 * P_discharge + 1.2292957997285587
        cooling_load = term1 - term2 *t_amb

        return -1 *cooling_load

    def charging_mfr(self,P_discharge):
        term2 = 0.0017455 * P_discharge -0.11658996400

        return term2

    def new_pressure(self,P_charge,P_discharge,P_tank,bin_charge,bin_discharge):
        timestep = 3600
        old_density = P_tank * 100000 / (287 * self.T_tank)
        new_density = (old_density * self.V_tank + (self.charging_mfr(P_charge)*bin_charge-
                                               self.discharging_mfr(P_discharge)*bin_discharge)*self.time_step)/self.V_tank

        new_pressure = new_density * (self.T_tank)*287 /100000   

        return new_pressure
    
    def update(self,P_charge, P_discharge):
        if P_charge >=80:
            self.pressure = self.new_pressure(P_charge,0,self.pressure,1,0)
            return self.pressure
        elif P_discharge >=80:
            self.pressure = self.new_pressure(0,P_discharge,self.pressure,0,1)
            return self.pressure
        else:
            self.pressure = self.new_pressure(0,0,self.pressure,0,0)
            return self.pressure


    def Charge(self, P_ch, T_amb):
      
        self.update(P_ch,0)
        
        self.SOC = round((self.pressure-self.P_min)/(self.P_max - self.P_min),ndigits=3)
        if P_ch>=80:
            P_ch_heat = P_ch * 0.38
        else:
            P_ch_heat = 0

        return (P_ch_heat)
  

    def Discharge(self,P_dch, T_amb):
        
        self.update(0,P_dch)
        
        self.SOC = round((self.pressure-self.P_min)/(self.P_max - self.P_min),ndigits=3)
        if P_dch>=80:
            P_dch_cool = self.cooling_load_predicted(P_dch,T_amb + 273)
        else:
            P_dch_cool = 0
        return P_dch_cool
      

#Defining Electricity and Gas Prices:
elec_price = 25*[0]
elec_price[0:7] = 7*[3.99]
elec_price[22] = 3.99
elec_price[23] = 3.99
elec_price[11:19] = 8*[19.9]
elec_price[7:11] = 4*[11.99]
elec_price[19:22] = 3*[11.99]
elec_price[24] = 3.99
gas_price = 3.21

#Defining The Environment
class Train_Env (gym.Env):
"""
  ###Decription
  
  The environment is consisted of Grid, WT, PV, MT, HP, AC, GB, CAES and loads.

  The state space is : Electricity Load, Heating Load, Cooling Load, PV Power, WT Power,SOC of AA-CAES, Buy Price Electricity, Ambient Temp, Timestep = 9 Elements
  
  The action space is : CAES, MT, HP, AC, GB, Heat/Cool Ratio of CAES

"""
  
  # Defining The Initial Parameters of Energy Hub
    def __init__ (self, p_grid_max=2000, p_caes_max=800, p_mt_max=2000,
                p_ec_max = 1000, p_hp_max = 1000, h_gb_max = 2000,
                n_mt = 0.35, h_ac_max = 1000, n_hr = 0.42, n_gb = 0.8, cop_hp = 3, cop_ac = 0.9,
                iterations = 24,c = 1000):

        self.penalty_coeff = c            #Penalty Coefficient
        self.iterations = iterations      #Iterations
        self.p_grid_max = p_grid_max      #Grid Max Power
        self.p_caes_max = p_caes_max      #CAES Max Power
        self.p_mt_max = p_mt_max          #MT Max Power
        self.p_hp_max = p_hp_max          #HP Max Power
        self.h_gb_max = h_gb_max          #GB Max Power
        self.h_ac_max = h_ac_max          #AC Max Power
        self.n_hr = n_hr                  #Heat Recovery Efficiency
        self.n_gb = n_gb                  #GB Efficiency
        self.cop_hp = cop_hp              #COP of HP
        self.cop_ac = cop_ac              #COP of AC
        self.time_step = 0                #Timestep
        self.n_mt = n_mt                  #MT Electric Efficiency
        self.op_cost = []                 #List for Operational Costs
        self.elecs_hist = [[],[],[],[]]   #Lists for Storing Electricity Components Powers
        self.heats_hist = [[],[],[],[],[]]#Lists for Storing Heating Components Powers
        self.colds_hist = [[],[],[]]      #Lists for Storing Cooling Components Powers
        self.picked_days = []             #List for Picked Random Days
        self.penalty_SOC_caes=[]          #List to Store the SOC Violation
        self.penalty_c_load = []          #List to Store Cooling Violation
        self.penalty_h_load = []          #List to Store Heating Violation
        self.penalty_p_grid=[]            #List to Store Grid Violation
        self.caes_SOC_hist = []           #List to Store SOC History
        self.penalty_cost_1 = []          #List to Store Penalty Cost 1
        self.penalty_cost_2 = []          #List to Store Penalty Cost 2
        self.penalty_cost_3 = []          #List to Store Penalty Cost 3
        self.penalty_cost_4 = []          #List to Store Penalty Cost 4
    

    #Defining State and Action Spaces:
    self.action_space = spaces.Box(low = np.array([-1,-1,0,0,0,0]), high = np.array([1,1,1,1,1,1]), dtype = np.float32,
                                   shape = (6,))
    self.observation_space = spaces.Box(low = -np.inf, high = np.inf, dtype=np.float32,
                                        shape = (9,))


  #Creating an AA-CAES System
    def _create_caes (self):
        self.caes = CAES_Class()
        self.initial_SOC_caes = self.caes.SOC
        return self.caes

    #Defining a Function to Output the Current State of The Environment
    def _build_state (self):
        
        SOC_caes = self.caes.SOC  

        self.wind_data = self.train["Wind Power"].reset_index(drop=True)
        wind = (self.wind_data[self.time_step] - min_wind)/ (max_wind- min_wind)

        self.temp = self.train["Temperature"].reset_index(drop=True)
        temp = (self.temp[self.time_step]- min_temp)/ (max_temp - min_temp)

        self.PV_data = self.train["PV Power"].reset_index(drop=True)
        pv = (self.PV_data[self.time_step]- min_pv)/ (max_pv - min_pv)

        self.elec_load = self.train["Electricity"].reset_index(drop=True)
        e_load = (self.elec_load[self.time_step]- min_elec)/ (max_elec - min_elec)

        self.heat_load = self.train["Heating"].reset_index(drop=True)
        h_load = (self.heat_load[self.time_step]- min_heat_load)/ (max_heat_load - min_heat_load)

        self.cold_load = self.train["Cooling"].reset_index(drop=True)
        c_load = (self.cold_load[self.time_step]- min_cold_load)/ (max_cold_load - min_cold_load)

        buy_price = (elec_price[self.time_step]- min(elec_price))/ (max (elec_price)- min(elec_price))

        state = np.array([SOC_caes, wind, pv, e_load, h_load, c_load, temp, buy_price, self.time_step/24])

        return state

    #Defining a Function to Calculate the Operational Cost of a timestep
    def operation_cost(self, p_grid, p_mt, h_gb):
        grid_prices_1 = 0.75 * elec_price[self.time_step]
        grid_prices_2 = 0.25 * elec_price[self.time_step]
        op_cost_grid = grid_prices_1 * p_grid + grid_prices_2 * abs(p_grid)
        op_cost_gb = h_gb/self.n_gb * gas_price
        op_cost_mt = (p_mt / self.n_mt) * gas_price

        return (op_cost_mt + op_cost_grid + op_cost_gb)/100
    
    #Defining Step Function for Performing an Action and Move to Next State
    def step(self, action):
        
        #Initializing reward and costs
        reward = 0
        cost_1 = 0
        cost_2 = 0
        cost_3 = 0
        cost_4 = 0

        #Reading the data from an action
        caes_action = action[0]
        hp_action = action[1]
        mt_action = action[2]
        ac_action = action[3]
        gb_action = action[4]
        caes_heat_cool_action = action[5]
        
        #Converting the action to power terms
        p_caes = self.p_caes_max * caes_action
        p_mt = self.p_mt_max * mt_action
        p_hp = self.p_hp_max * hp_action
        h_hr = p_mt * (self.n_hr / self.n_mt)
        h_ac = ac_action * self.h_ac_max
        c_ac = h_ac * self.cop_ac
        h_gb = gb_action * self.h_gb_max
        
        #Making Sure that Power of MT is less than its Minimum value = 200
        if p_mt<199.9999:
            p_mt=0
            h_hr=0
        
        #Making Sure that Power of CAES is less than its Minimum value = 80:
        if abs(p_caes)<79.999:
            p_caes = 0

        #CAES Charging or Discharging:
        if (p_caes > 0):
            c_caes = self.caes.Discharge(p_caes,self.temp[self.time_step]) * caes_heat_cool_action

        if c_caes <0:
            h_caes = abs(c_caes) * caes_heat_cool_action
            c_caes = 0

        else:
            h_caes = 0


        if (p_caes < 0):
            h_caes = self.caes.Charge(-p_caes,self.temp[self.time_step]) * caes_heat_cool_action
            c_caes = 0

        else:
            c_caes = 0
            h_caes = 0
            pass

        #Heat Pump Operation COoling or Heating:
        if p_hp <0:
            c_hp = abs(p_hp) * self.cop_hp
            h_hp =0
        else:
            h_hp = p_hp * self.cop_hp
            c_hp = 0
    
        
        #Heating and Cooling Power Generation:
        cool_generated = c_hp + c_caes + c_ac

        heat_generated = h_gb + h_hr + h_hp + h_caes
        
        #Calculating Power Exchanged With Grid:
        p_grid = self.elec_load[self.time_step] + abs(p_hp) - (self.wind_data[self.time_step] + self.PV_data[self.time_step]+ p_caes + p_mt)


        #Creating Dataframes of Powers
        self.elecs_hist[0].append(p_grid)
        self.elecs_hist[1].append(p_caes)
        self.elecs_hist[2].append(p_mt)
        self.elecs_hist[3].append(abs(p_hp))


        self.heats_hist[0].append(h_gb)
        self.heats_hist[1].append(h_hr)
        self.heats_hist[2].append(h_hp)
        self.heats_hist[3].append(h_ac)
        self.heats_hist[4].append(h_caes)


        self.colds_hist[0].append(c_ac)
        self.colds_hist[1].append(c_hp)
        self.colds_hist[2].append(c_caes)

        self.caes_SOC_hist.append(self.caes.SOC)


        # Adding operation cost to reward function
        reward -= self.operation_cost(p_grid, p_mt, h_gb)
        self.op_cost.append(self.operation_cost(p_grid, p_mt, h_gb))

        #Constraints Calculation
        u_caes_up = 0
        u_caes_low = 0
        u_grid_up = 0
        u_grid_low = 0

        if self.caes.SOC < 0.2:
            u_caes_low = 1

        elif self.caes.SOC > 0.9:
            u_caes_up = 1

        if p_grid > self.p_grid_max:
            u_grid_up = 1

        elif p_grid < -self.p_grid_max:
            u_grid_low = 1

        penalty_caes = (abs(self.caes.SOC - 0.9)*u_caes_up + abs(self.caes.SOC - 0.2)*u_caes_low)*self.penalty_coeff
        penalty_grid = (abs(p_grid - self.p_grid_max)*u_grid_up + abs(p_grid + self.p_grid_max)*u_grid_low)
        penalty_h_load = abs (self.heat_load[self.time_step]- heat_generated) 
        penalty_c_load = abs (self.cold_load[self.time_step] - (cool_generated))
        
        penalty_cost = penalty_grid + penalty_caes + penalty_h_load + penalty_c_load

        self.penalty_costs.append(penalty_cost)
        self.penalty_SOC_caes.append(penalty_caes)
        self.penalty_p_grid.append(penalty_grid)
        self.penalty_c_load.append(penalty_c_load)
        self.penalty_h_load.append(penalty_h_load)

        cost_1 += penalty_caes
        cost_2 += penalty_grid
        cost_3 += penalty_c_load
        cost_4 += penalty_h_load

        self.penalty_cost_1.append(penalty_caes)
        self.penalty_cost_2.append(penalty_grid)
        self.penalty_cost_3.append(penalty_c_load)
        self.penalty_cost_4.append(penalty_h_load)
    
        #Moving to next state
        self.time_step += 1
        done = self.time_step == 24
        
        #Final SOC of AA-CAES Constraint
        if done:
            penalt = abs(self.caes.SOC - 0.5) *self.penalty_coeff
            if penalt <1e-3:
                penalt = 0
            reward -= penalt
        
        #Build the Next State of The Environment
        state = self._build_state()
        info = None

        return state, reward, cost_1, cost_2, cost_3, cost_4, done, info 

    #Defining a Funciton to reset the Environment at the End of a Day
    def reset (self):

        self.time_step = 0
        self.caes = self._create_caes()
        self.op_cost = []
        self.penalty_costs = []
        self.penalty_cost_1 = []
        self.penalty_cost_2 = []
        self.penalty_cost_3 = []
        self.penalty_cost_4 = []

        self.elecs_hist = [[],[],[],[]]
        self.heats_hist = [[],[],[],[],[]]
        self.colds_hist = [[],[],[]]
        self.penalty_SOC_caes = []
        self.penalty_h_load = []
        self.penalty_gas_consump = []
        self.penalty_p_grid = []
        self.penalty_c_load = []
        self.caes_SOC_hist = []
        self.get_day()
        return self._build_state()


    #Getting the Information of a Random Day in Dataset
    def get_day (self):
        if len(self.picked_days)==len(days_to_train):
            self.picked_days = list()

        available_days = [day for day in list(days_to_train) if day not in self.picked_days ]
        day = available_days[0]

        if day == 365:
            self.train = (data[data["Day"]==day]).append(all_data[all_data["Day"]==1].iloc[0])

        elif day+1 not in list(days_to_train):
            self.train = (data[data["Day"]==day]).append(all_data[all_data["Day"]==(day+1)].iloc[0])
        else:
            self.train = (data[data["Day"]==day]).append(data[data["Day"]==(day+1)].iloc[0])

        self.picked_days.append(day)
        return self.train

    #Defining a Random Seed For Environment
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
