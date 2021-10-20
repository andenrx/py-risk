import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import json
from game_types import MapState

def load_data(directory, mapstruct):
    data = []
    for file in os.listdir(directory):
        data += json.load(open(directory + file))["turns"]
    states = [MapState(entry["armies"], entry["owner"], mapstruct) for entry in data]
    return [
        (
            state.to_tensor(1, 2),
            (entry["p1_win_value"] / entry["p1_visits"],
            -entry["p2_win_value"] / entry["p2_visits"])
        )
        for state, entry in zip(states, data)
    ]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GCNConv(3, 5)
        self.layer2 = GCNConv(5, 5)
        self.layer3 = GCNConv(5, 1)
        self.dense1 = Linear(5, 15)
        self.dense2 = Linear(15, 1)
 
    def forward(self, x1, x2, edges):
        x = self.layer1(x1, edges)
        x = F.relu(x)
        x = self.layer2(x, edges)
        x = F.relu(x)
        x = self.layer3(x, edges)
        x = x.mean()
        x = torch.cat([torch.tensor([x]), x2])
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)[0]
        return torch.tanh(x)

def train_model(model, opt):
    for i in range(50):
        opt.zero_grad()
        loss = 0
        for (x1, x2, edges), (y1, y2) in train:
            out = model(x1, x2, edges)
            loss += F.mse_loss(out, y1)
            loss += F.mse_loss(out, y2)
        loss.backward()
        opt.step()

class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GCNConv(16, 8)
        self.layer2 = GCNConv(8, 8)
        self.layer3 = GCNConv(8, 5)
        self.dense1 = Linear(11, 15)
        self.dense2 = Linear(15, 1)

    def forward(self, x1, x2, edges):
        x = self.layer1(x1, edges)
        x = F.relu(x)
        x = self.layer2(x, edges)
        x = F.relu(x)
        x = self.layer3(x, edges)
        x = torch.cat([x.mean(axis=0), x2])
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)[0]
        return torch.tanh(x)

class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(15, 15)
        self.layer2 = GCNConv(34, 15)
        self.layer3 = Linear(34, 1)
        self.layer4 = Linear(19, 19)

    def forward(self, x1, x2, edges):
      # combine globals with graph someohow?
      x2 = torch.tile(x2, (20,1,))
      b = self.layer4(torch.cat([x1, x2], axis=1))
      a = F.relu(self.layer1(x1))
      for i in range(5):
        a = F.relu(self.layer2(torch.cat([a, b], axis=1), edges))
      a = self.layer3(torch.cat([a, b], axis=1))
      return torch.tanh(a.mean())

