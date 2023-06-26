#Importing Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Reading Optimal Answers For Training Days:
optimal_answers = pd.read_csv("Training_answers.csv")

#Electricity and Gas Price:
buy_price = 24*[0]
buy_price[0:7] = 7*[3.99]
buy_price[22] = 3.99
buy_price[23] = 3.99
buy_price[11:19] = 8*[19.9]
buy_price[7:11] = 4*[11.99]
buy_price[19:22] = 3*[11.99]
gas_price = 3.21

#Defining a class for Actor Network:

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 2)
        self.layer_4 = nn.Linear(128, action_dim-2)
   
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x1 = self.max_action * torch.tanh(self.layer_3(x))
        x2 = self.max_action * torch.sigmoid(self.layer_4(x))
        x = torch.cat([x1,x2],1)
        return x

#Defining the Actions from Optimal Answers:
y_true = optimal_answers[["CAES","P_hp","P_gt","H_ac","H_gb","caes_heat_cool"]]

#Normalizing the actions based on maximum capacity of each equipment:
y_true["CAES"] = y_true["CAES"]/800
y_true["P_hp"] = y_true["P_hp"]/1000
y_true["P_gt"] = y_true["P_gt"]/1500
y_true["H_ac"] = y_true["H_ac"]/(1000/0.9)
y_true["H_gb"] = y_true["H_gb"]/1500

#Defining the input features (States of the Problem) and also scale them:

x = pd.DataFrame()

x["SOC_CAES"] = (optimal_answers["Pressure_CAES"] - 42)/30

x["Wind"] = (optimal_answers["Wind Power"] - min(optimal_answers["Wind Power"])) / (max(optimal_answers["Wind Power"])- min(optimal_answers["Wind Power"]))

x["PV"] = (optimal_answers["PV Power"] - min(optimal_answers["PV Power"])) / (max(optimal_answers["PV Power"])- min(optimal_answers["PV Power"]))

x["Electricity"] = (optimal_answers["Electricity"] - min(optimal_answers["Electricity"])) / (max(optimal_answers["Electricity"])- min(optimal_answers["Electricity"]))

x["Heating"] = (optimal_answers["Heating"] - min(optimal_answers["Heating"])) / (max(optimal_answers["Heating"])- min(optimal_answers["Heating"]))

x["Cooling"] = (optimal_answers["Cooling"] - min(optimal_answers["Cooling"])) / (max(optimal_answers["Cooling"])- min(optimal_answers["Cooling"]))

x["Temperature"] = (optimal_answers["Temperature"] - min(optimal_answers["Temperature"])) / (max(optimal_answers["Temperature"])- min(optimal_answers["Temperature"]))

x["Buy Price"] = (buy_price * 365) 
x["Buy Price"] = [(i - (min(buy_price))) / (max(buy_price) - min(buy_price)) for i in x["Buy Price"].values]

x["Time Step"] = optimal_answers["timestep"] / 24

#Converting X and Y to Tensors:
x_train = torch.tensor(x.values).float()
y_train = torch.tensor(y_true.values).float()

#Creating a Dataset From Tensors:
from torch.utils.data import TensorDataset
dataset = TensorDataset(x_train, y_train)

#Defining MSE Loss and Random batching:
from torch.utils.data import DataLoader

batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def mse_loss(predictions, targets):
    difference = predictions - targets
    return torch.sum(difference * difference)/ difference.numel()


#Creating an Actor Network:
actor = Actor(9,6,1)

#Defining The Adam Optimizer:
optimizer = torch.optim.Adam(params = actor.parameters(),lr=0.0001)

#Saving The Loss Functions:
loss_hist=[]


#Starting Learning Process:
epochs = 1000
from math import sqrt
for i in range(epochs):
    # Iterate through training dataloader
    for x,y in train_loader:
        # Generate Prediction
        preds = actor(x)
        # Get the loss and perform backpropagation
        loss = mse_loss(preds, y)
        optimizer.zero_grad()
        loss.backward()
        # Let's update the weights
        optimizer.step()
        loss_hist.append((loss.detach().item()))
    if loss.detach().item()<0.007: #Stop If loss is lower than 0.007
        break
    print(f"Epoch {i}/{epochs}: Loss: {(loss)}")
    
    
#Saving The Learned Actor Network (Behavioral Cloning Network):
torch.save(actor.state_dict(), 'pretrained_actor.pth')

