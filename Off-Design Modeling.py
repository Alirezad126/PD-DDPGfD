#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#AA-CAES Specifications:
P_CAES_ch_max = 800    #Maximum Charging Power
P_CAES_dch_max = 800   #Maximum Discharging Power
gamma = 1.4            #Air Isentropic Expansion Factor
cp = 1.005             #Air Specific Heat Capacity
T_amb = 293            #Ambient Air Temperature
T_tank = 293           #Air Tank Temperature
V_tank = 1100.19       #Air Tank Volume
timestep = 3600        #1-hour Timestep
eff = 0.9              #Heat Exchanger Effectiveness
T_cold_w = 273+20      #Cold Water Tank Temperature
T_hot_w = 273+95       #Hot Water Tank Temperature

#Functions For Operation:

#Off-desing Curves:
def P_ratio_exp_1_offdesign (P_discharge):
    a1 = 1.0526686612247751e+000
    a2 = 3.3837646597101853e-002
    a3 = -1.0594350616851346e-004
    a4 = 4.3037073018992756e-007
    a = P_discharge/P_CAES_dch_max* 100
    HPT_R = a1 + a2 * (a) + a3 * (a**2) + a4 * (a**3)
    return HPT_R

def P_ratio_exp_2_offdesign (P_discharge):
    a1 = 1.8675819367444066e+000
    a2 = 1.1792419504272698e-001
    a3 = -5.0802207851639934e-004
    a4 = 1.9700035951777700e-006
    a = P_discharge/P_CAES_dch_max * 100
    LPT_R = a1 + a2 * (a) + a3 * (a**2) + a4 * (a**3)
    return LPT_R

def Isentropic_comp_offdesign (P_charge):
    a1 =  4.6948804252492010e+001
    a2 =  5.9916135114652325e-001
    a3 = -2.3775603099293894e-003
    a = P_charge/P_CAES_ch_max* 100
    eff_comp = a1 + a2 * (a) + a3 * (a**2)
    return (eff_comp/100)

def Isentropic_exp_offdesign (P_discharge):
    a1 = 4.1390729581326283e+001
    a2 = 8.6154009677822008e-001
    a3 = -5.7158107778840558e-003
    a4 = 9.5398673887267854e-006
    a = P_discharge/P_CAES_dch_max* 100
    eff_exp = a1 + a2 * (a) + a3 * (a**2) + a4 * (a**3)
    return (eff_exp/100)


#Outlet Temperature of Compressor 1:
def T_out_comp_1_offdesign (P_charge):
    
    T_out_comp_1 = ((T_amb) * (1 + (((P_ratio_comp_1_offdesign(P_tank))**((gamma-1)/gamma))-1)/Isentropic_comp_offdesign(P_charge)))
    return T_out_comp_1

#Outlet Temperature of Compressor 2:
def T_out_comp_2_offdesign (P_charge):
    T_out_HE1 = T_out_HE1_offdesign(P_charge,P_tank)
    T_out_comp_2 = ((T_out_HE1) * (1 + (((P_ratio_comp_2_offdesign(P_tank))**((gamma-1)/gamma))-1)/Isentropic_comp_offdesign(P_charge)))

    return T_out_comp_2

#Outlet Temperature of Heat Exchanger 1:
def T_out_HE1_offdesign (P_charge,P_tank):
    T_out_HE1 = T_out_comp_1_offdesign (P_charge, P_tank) * (1-eff) + T_cold_w * eff

    return T_out_HE1

#Charging Mass Flow Rate Off Design:
def charging_mfr_offdesign (P_charge, P_tank): 
    work = cp * (T_out_comp_1_offdesign(P_charge, P_tank) - T_amb +  T_out_comp_2_offdesign(P_charge,P_tank) -
                         T_out_HE1_offdesign (P_charge,P_tank))
    charging_mfr = P_charge/work
    return charging_mfr

#Charging Heat Generation Off Design:
def charging_heat_offdesign (P_charge, P_tank): 
    heat = charging_mfr_offdesign (P_charge, P_tank) * (T_out_comp_1_offdesign(P_charge, P_tank) - T_out_HE1_offdesign (P_charge,P_tank))
    return heat

#Outlet Temperature of Expander 1:
def T_out_exp_1_offdesign (P_discharge):
    T_out_HE3 = T_out_HE3_offdesign (P_discharge)
    T_out_exp_1 = ((T_out_HE3) * (1 - (Isentropic_exp_offdesign(P_discharge) *
                                       (1 - ((1/P_ratio_exp_1_offdesign(P_discharge))**((gamma-1)/gamma))))))
    return T_out_exp_1

#Outlet Temperature of Expander 2:
def T_out_exp_2_offdesign (P_discharge):
    T_out_HE4 = T_out_HE4_offdesign (P_discharge) 
    T_out_exp_2 = ((T_out_HE4) * (1 - (Isentropic_exp_offdesign(P_discharge) *
                                       (1 - ((1/P_ratio_exp_2_offdesign(P_discharge))**((gamma-1)/gamma))))))
    return T_out_exp_2

#Outlet Temperature of Heat Exchanger 3:
def T_out_HE3_offdesign (P_discharge):
    T_out_HE3 = T_amb * (1-eff) + T_hot_w * eff

    return T_out_HE3

#Outlet Temperature of Heat Exchanger 4:
def T_out_HE4_offdesign (P_discharge):
    T_out_HE4 = T_out_exp_1_offdesign (P_discharge) * (1-eff) + T_hot_w * eff

    return T_out_HE4

#Updating the Pressure of Air Storage:
def new_pressure_offdesign(P_charge,P_discharge,P_tank):
    timestep = 3600
    old_density = P_tank * 100000 / (287 * T_tank)
    new_density = (old_density * V_tank + (charging_mfr_offdesign(P_charge,P_tank)-
                                           discharging_mfr_offdesign(P_discharge))*timestep)/V_tank
    
    new_pressure = new_density * (T_tank)*287 /100000   
    
    return new_pressure

#Discharging Mass Flow Rate Off Design:
def discharging_mfr_offdesign(P_discharge):
    output_work = cp * (T_out_HE3_offdesign (P_discharge) - T_out_exp_1_offdesign(P_discharge) +  T_out_HE4_offdesign (P_discharge) -T_out_exp_2_offdesign(P_discharge))
    
    discharging_mfr = P_discharge/output_work
    
    return discharging_mfr
    
#Discharging Cooling Generation Off Design:
def discharging_cool_offdesign(P_discharge):
    cool = discharging_mfr_offdesign(P_discharge) * cp * (T_out_exp_2_offdesign(P_discharge))
    return cool

#Extra Functions For Linearizing Cooling Power Generation:
def discharging_ambient_cool_offdesign(P_discharge):
    cool = discharging_mfr_offdesign(P_discharge) * cp
    return cool

def cooling_load_off_design(P_discharge,t_amb):
    cool = discharging_mfr_offdesign(P_discharge) * cp * (t_amb - T_out_exp_2_offdesign(P_discharge))
    return cool



#Linearizing The Operation:

# x is the charing and discharging power nad y is the mass flow rate
x = np.arange(80,800.1,0.1)
y1 = list()
for i in x:
    y1.append(discharging_mfr(i))
    
y2 = list()
for i in x:
    y2.append(discharging_mfr_offdesign(i))
    
x_ratio = [i/P_CAES_ch_max *100 for i in x]


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=1, include_bias=False)
poly_features = poly.fit_transform(x.reshape(-1, 1))
from sklearn.linear_model import LinearRegression
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y2)
poly_reg_y_predicted_lin = poly_reg_model.predict(poly_features)
from sklearn.metrics import mean_squared_error
poly_reg_rmse_lin = np.sqrt(mean_squared_error(y2, poly_reg_y_predicted_lin))

#Reading coefficient and intercept of linear regression
coeff = poly_reg_model.coef_ 
intercept = poly_reg_model.intercept_ 