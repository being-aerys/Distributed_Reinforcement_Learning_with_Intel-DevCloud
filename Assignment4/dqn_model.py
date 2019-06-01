import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

FloatTensor = torch.FloatTensor

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
        module.bias.data.fill_(0.01)
        
class _DQNModel(nn.Module):
    """ Model for DQN """

    def __init__(self, input_len, output_len):
        super(_DQNModel, self).__init__()
        
        self.fc1 = nn.Sequential(
            torch.nn.Linear(input_len, 256),
            nn.Tanh()
        )
        self.fc1.apply(weights_initialize)
        
        self.fc2 = nn.Sequential(
            torch.nn.Linear(256, 64),
            nn.Tanh()
        )
        self.fc2.apply(weights_initialize)
        
        self.output_layer = nn.Sequential(
            torch.nn.Linear(64, output_len)
        )
        self.output_layer.apply(weights_initialize)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        
        return self.output_layer(x)

    
class DQNModel():
    def __init__(self, input_len, ouput_len, learning_rate = 0.0005):
        self.model = _DQNModel(input_len, ouput_len)
       
        self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.model.parameters(), lr = learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.steps = 0
        
    def predict(self, input):
        input = FloatTensor(input).unsqueeze(0)
        q_values = self.model(input)
        action = torch.argmax(q_values)

        return action.item()
    
    def predict_batch(self, input):
        input = FloatTensor(input)
        q_values = self.model(input)
        values, q_actions = q_values.max(1)
        
        return q_actions, q_values

    def fit(self, q_values, target_q_values):
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        
    def replace(self, dest):
        self.model.load_state_dict(dest.model.state_dict())
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))