#Importing Libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Using Cuda if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Defining Actor, Critic, and Cost Networks:
#Actor Network:
class Actor(nn.Module):
    
    #Initial Layers and Dimentions:
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)     #First Layer
        self.layer_2 = nn.Linear(256, 128)           #Second Layer
        self.layer_3 = nn.Linear(128,2)              #Third Layer for Those actions between [-1,1]
        self.layer_4 = nn.Linear(128, action_dim-2)  #Third Layer for Those actions between [0,1]
        self.max_action = max_action
    
    #Forward Propagation:
    def forward(self, x):
        x = F.relu(self.layer_1(x))                             #Forward Passing With ReLU Function Layer 1
        x = F.relu(self.layer_2(x))                             #Forward Passing With ReLU Function Layer 2
        x1 = self.max_action * torch.tanh(self.layer_3(x))      #Forward Passing With Tanh Function for [-1,1]
        x2 = self.max_action * torch.sigmoid(self.layer_4(x))   #Forward Passing With ReLU Function for [0,1]
        x = torch.cat([x1,x2],1)                                #Concatenating Layer 3 and 4 for the Action Output
        return x

#Critic Network:
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #Defining the critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 1)


    def forward(self, x, u):
        xu = torch.cat([x,u], 1)
        #forward propagation on the first critic neural network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

    def Q1(self, x, u):
        xu = torch.cat([x,u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

#Cost Network:
class Cost(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Cost, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 1)


    def forward(self, x, u):
        xu = torch.cat([x,u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

    def C1(self, x, u):
        xu = torch.cat([x,u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
    

#Defining PD-DDPGfD Algorithm:
class PD_DDPGfD(object):

    #Initializing the network parameters:
    def __init__(self, state_dim, action_dim, max_action, lambda_, lambda_step, constraint_limit_1, constraint_limit_2,
                 constraint_limit_3, constraint_limit_4,lr_critics, lr_actor):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)                  #Creating an Actor Model
        self.actor.load_state_dict(torch.load('../Pre-training/pretrained_actor.pth'))    #Loading the Pre-trained Model
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)           #Creating an Actor-target Model
        self.actor_target.load_state_dict(self.actor.state_dict())                        #Copying Actor to Actor-target
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)     #Defining Optimizer for Actor Model

        self.critic = Critic(state_dim, action_dim).to(device)                            #Creating a Critic Model
        self.critic.load_state_dict(torch.load('pretrained_critic.pth'))                  #Loading Pre-trained Critic Model
        self.critic_target = Critic(state_dim, action_dim).to(device)                     #Creating Critic-target Model
        self.critic_target.load_state_dict(self.critic.state_dict())                      #Copying Critic to Critic-target
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critics) #Defining Optimizer for Actor Model
        
        #Cost Networks are similar to Critic networks:
        self.cost_1 = Cost(state_dim, action_dim).to(device)
        self.cost_1.load_state_dict(torch.load('pretrained_cost_1.pth'))
        self.cost_1_target = Cost(state_dim, action_dim).to(device)
        self.cost_1_target.load_state_dict(self.cost_1.state_dict())
        self.cost_1_optimizer = torch.optim.Adam(self.cost_1.parameters(), lr=lr_critics)

        self.cost_2 = Cost(state_dim, action_dim).to(device)
        self.cost_2.load_state_dict(torch.load('pretrained_cost_2.pth'))
        self.cost_2_target = Cost(state_dim, action_dim).to(device)
        self.cost_2_target.load_state_dict(self.cost_2.state_dict())
        self.cost_2_optimizer = torch.optim.Adam(self.cost_2.parameters(), lr=lr_critics)

        self.cost_3 = Cost(state_dim, action_dim).to(device)
        self.cost_3.load_state_dict(torch.load('pretrained_cost_3.pth'))
        self.cost_3_target = Cost(state_dim, action_dim).to(device)
        self.cost_3_target.load_state_dict(self.cost_3.state_dict())
        self.cost_3_optimizer = torch.optim.Adam(self.cost_3.parameters(), lr=lr_critics)

        self.cost_4 = Cost(state_dim, action_dim).to(device)
        self.cost_4.load_state_dict(torch.load('pretrained_cost_4.pth'))
        self.cost_4_target = Cost(state_dim, action_dim).to(device)
        self.cost_4_target.load_state_dict(self.cost_4.state_dict())
        self.cost_4_optimizer = torch.optim.Adam(self.cost_4.parameters(), lr=lr_critics)

        self.max_action = max_action
        
        #Defining dual-variables for each cost network
        self.lambda_1 = torch.tensor([lambda_], requires_grad=False).to(device)
        self.lambda_2 = torch.tensor([lambda_], requires_grad=False).to(device)
        self.lambda_3 = torch.tensor([lambda_], requires_grad=False).to(device)
        self.lambda_4 = torch.tensor([lambda_], requires_grad=False).to(device)

        #Defining dual-variables learning rate:
        self.lambda_step = lambda_step
        
        #defining constraint toleration for each constraint:
        self.constraint_limit_1 = constraint_limit_1
        self.constraint_limit_2 = constraint_limit_2
        self.constraint_limit_3 = constraint_limit_3
        self.constraint_limit_4 = constraint_limit_4
        
        #Storing Loss For Each Network:
        self.Q_loss = []
        self.C1_loss = []
        self.C2_loss = []
        self.C3_loss = []
        self.C4_loss = []
        self.C4_value = []
        self.actor_losss = []
        self.Q_value = []
        self.Q_value_target = []

    #Defining a function for output Action for a given State
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    #Defining a function for training process:
    def train(self, replaybuffer_pretrain, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 1: We sample a batch of transitions (s, s', a, r, c1, c2, c3, c4) from the memory
            batch_states, batch_next_states, batch_actions, batch_next_actions, batch_rewards,
            batch_costs_1, batch_costs_2, batch_costs_3, batch_costs_4, batch_dones = replaybuffer_pretrain.sample(batch_size)
            
            # Step 2: Converting numpy arrays to tensors
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            cost_1 = torch.Tensor(batch_costs_1).to(device)
            cost_2 = torch.Tensor(batch_costs_2).to(device)
            cost_3 = torch.Tensor(batch_costs_3).to(device)
            cost_4 = torch.Tensor(batch_costs_4).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 3: From the next state s', the Actor target plays the next action a'
            next_action = self.actor_target(next_state)

            # Step 4: The  Critic and Cost targets take each the couple (s', a') as input and return two Q-values Qt1(s',a') and Ct(s',a') as outputs:
            target_Q = self.critic_target(next_state, next_action)
            target_C_1 = self.cost_1_target(next_state, next_action)
            target_C_2 = self.cost_2_target(next_state, next_action)
            target_C_3 = self.cost_3_target(next_state, next_action)
            target_C_4 = self.cost_4_target(next_state, next_action)

            # Step 5: We get the final target of the Critic and Cost models, which is: Qt = r + gamma * min(Qt1, Qt2), where gamma is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            target_C_1 = cost_1 + ((1 - done) * discount * target_C_1).detach()
            target_C_2 = cost_2 + ((1 - done) * discount * target_C_2).detach()
            target_C_3 = cost_3 + ((1 - done) * discount * target_C_3).detach()
            target_C_4 = cost_4 + ((1 - done) * discount * target_C_4).detach()

            # Step 6: The Critic model takes the couple (s, a) as input and return two Q-values Q1(s,a) as output
            current_Q = self.critic(state, action)
            self.Q_value.append(current_Q.sum().detach())
            self.Q_value_target.append(target_Q.sum().detach())

            # Step 7: We compute the loss coming from the Critic model: Critic Loss = MSE_Loss(Q(s,a), Qt)
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Step 8: We backpropagate this Critic loss and update the parameters of Critic model with an Adam optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.Q_loss.append(critic_loss.detach())

            # Step 9: The Cost models take each the couple (s, a) as input and return two C-values C(s,a) as outputs
            current_C_1 = self.cost_1(state, action)
            current_C_2 = self.cost_2(state, action)
            current_C_3 = self.cost_3(state, action)
            current_C_4 = self.cost_4(state, action)

            # Step 10: We compute the loss coming from the Cost models: Cost Loss = MSE_Loss(C(s,a), Ct) for each network:
            cost_loss_1 = F.mse_loss(current_C_1, target_C_1)
            self.cost_1_optimizer.zero_grad()
            cost_loss_1.backward()
            self.cost_1_optimizer.step()

            cost_loss_2 = F.mse_loss(current_C_2, target_C_2)
            self.cost_2_optimizer.zero_grad()
            cost_loss_2.backward()
            self.cost_2_optimizer.step()

            cost_loss_3 = F.mse_loss(current_C_3, target_C_3)
            self.cost_3_optimizer.zero_grad()
            cost_loss_3.backward()
            self.cost_3_optimizer.step()

            cost_loss_4 = F.mse_loss(current_C_4, target_C_4)
            self.cost_4_optimizer.zero_grad()
            cost_loss_4.backward()
            self.cost_4_optimizer.step()

            #Storing C Loss:
            self.C1_loss.append(cost_loss_1.detach())
            self.C2_loss.append(cost_loss_2.detach())
            self.C3_loss.append(cost_loss_3.detach())
            self.C4_loss.append(cost_loss_4.detach())

            # Step 11: Once every some iterations, we update our Actor model by performing gradient ascent on the output of the Critic and Cost models:
            if it % policy_freq == 0:
                Q1 = self.critic.Q1(state, self.actor(state))
                C1 = self.cost_1.C1(state, self.actor(state))
                C2 = self.cost_2.C1(state, self.actor(state))
                C3 = self.cost_3.C1(state, self.actor(state))
                C4 = self.cost_4.C1(state, self.actor(state))
                
                #Adding an extra term for imitating the expert demonstrations:
                action_loss = F.mse_loss(action, self.actor(state))

                actor_loss = - (Q1 - self.lambda_1.detach() * C1 - self.lambda_2.detach() * C2 - self.lambda_3.detach() * C3 - self.lambda_4.detach() * C4 - action_loss).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Step 12: Still once every some iterations, we update the weights of the Actor target by polyak averaging and dual-variables:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                lambda_gradient_1 = (self.cost_1.C1(state, self.actor(state)) - self.constraint_limit_1).mean()
                self.lambda_1 = max(torch.tensor([0]).to(device),
                                    self.lambda_1 + self.lambda_step * lambda_gradient_1).detach()

                lambda_gradient_2 = (self.cost_2.C1(state, self.actor(state)) - self.constraint_limit_2).mean()
                self.lambda_2 = max(torch.tensor([0]).to(device),
                                    self.lambda_2 + self.lambda_step * lambda_gradient_2).detach()

                lambda_gradient_3 = (self.cost_3.C1(state, self.actor(state)) - self.constraint_limit_3).mean()
                self.lambda_3 = max(torch.tensor([0]).to(device),
                                    self.lambda_3 + self.lambda_step * lambda_gradient_3).detach()

                lambda_gradient_4 = (self.cost_4.C1(state, self.actor(state)) - self.constraint_limit_4).mean()
                self.lambda_4 = max(torch.tensor([0]).to(device),
                                    self.lambda_4 + self.lambda_step * lambda_gradient_4).detach()

                # Step 13: Still once every some iterations, we update the weights of the Critic and Costs target by polyak averaging:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.cost_1.parameters(), self.cost_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.cost_2.parameters(), self.cost_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.cost_3.parameters(), self.cost_3_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.cost_4.parameters(), self.cost_4_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.cost_1.state_dict(), '%s/%s_cost_1.pth' % (directory, filename))
        torch.save(self.cost_2.state_dict(), '%s/%s_cost_2.pth' % (directory, filename))
        torch.save(self.cost_3.state_dict(), '%s/%s_cost_3.pth' % (directory, filename))
        torch.save(self.cost_4.state_dict(), '%s/%s_cost_4.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.cost_1.load_state_dict(torch.load('%s/%s_cost_1.pth' % (directory, filename)))
        self.cost_2.load_state_dict(torch.load('%s/%s_cost_2.pth' % (directory, filename)))
        self.cost_3.load_state_dict(torch.load('%s/%s_cost_3.pth' % (directory, filename)))
        self.cost_4.load_state_dict(torch.load('%s/%s_cost_4.pth' % (directory, filename)))
