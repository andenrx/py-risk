import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import json
from .game_types import MapState
from .orders import *

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
    def predict_policy(self): return True

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
    def predict_policy(self): return True
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
    def predict_policy(self): return True
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

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv, GATv2Conv

class Model5(torch.nn.Module):
    def predict_policy(self): return False
    def __init__(self):
        super().__init__()

        self.g1 = GATv2Conv(15, 10, dropout=0.25)
        self.g2 = GATv2Conv(10 + 15, 10, dropout=0.25)
        self.g3 = GATv2Conv(10 + 15, 10, dropout=0.25)

        self.lin = Linear(10 + 15 + 4, 15)
        self.lin2 = Linear(15, 1)

        self.aaa = Linear(20+30+2-4, 20)
        self.bbb = Linear(20, 1)

        self.ccc = Linear(10+15+1-3, 20)
        self.ddd = Linear(20, 1)

        self.drop = Dropout(0.25)

    def forward(self, x1, x2, edges, moves):
      x2 = torch.tile(x2, (x1.size()[0], 1))
      x = torch.cat([x1, x2], dim=1)

      x = self.g1(x1, edges)
      x = F.relu(x)
      x = self.g2(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)
      x = self.g3(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)

      p = torch.zeros(len(moves))
      for i, move in enumerate(moves):
        for j, m in enumerate(move):
          if isinstance(m, AttackTransferOrder):
            asdf = self.aaa(torch.cat([
                    x[m.src, :],
                    x[m.dst, :],
                    x1[m.src, 3:],
                    x1[m.dst, 1:],
                    torch.tensor([
                        m.armies,
                        0.6 * m.armies - 0.7 * (x1[m.dst, 3] + x1[m.dst, 4])
                        ])]))# 0.6 attackers - 0.7 defenders
            asdf = self.bbb(F.relu(self.drop(asdf)))
            p[i] += asdf[0]
          elif isinstance(m, DeployOrder):
            qwer = self.ccc(torch.cat([
                x[m.target, :], x1[m.target, 3:], torch.tensor([m.armies])
            ]))
            qwer = self.ddd(F.relu(self.drop(qwer)))
            p[i] += qwer[0]
          else:
            raise Exception(m)

      V = self.lin(torch.cat([self.drop(x), x1, x2], dim=1))
      V = self.lin2(F.relu(self.drop(V)))
      V = torch.tanh(V.mean())

      return V, F.log_softmax(p, dim=0)

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv, GATv2Conv

class Model6(torch.nn.Module):
    def predict_policy(self): return False
    def __init__(self):
        super().__init__()
        self.g1 = GATv2Conv(15, 10, dropout=0.25)
        self.g2 = GATv2Conv(10 + 15, 10, dropout=0.25)
        self.g3 = GATv2Conv(10 + 15, 10, dropout=0.25)

        self.lin = Linear(10 + 15 + 4, 15)
        self.lin2 = Linear(15, 1)

        self.aaa = Linear(20+30+2-4, 20)
        # self.bbb = Linear(20, 20)

        self.ccc = Linear(10+15+1-3, 20)
        # self.ddd = Linear(20, 20)

        self.att_layer = Linear(20, 1)
        self.pi_layer = Linear(20, 1)

        self.drop = Dropout(0.25)

    def forward(self, x1, x2, edges, moves):
      x2 = torch.tile(x2, (x1.size()[0], 1))
      x = torch.cat([x1, x2], dim=1)

      x = self.g1(x1, edges)
      x = F.relu(x)
      x = self.g2(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)
      x = self.g3(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)

      # torch.softmax(Q @ (K @ Wk).T / torch.sqrt(torch.tensor(5)), dim=0) @ (V @ Wv)
      p = torch.zeros(len(moves))
      for i, move in enumerate(moves):
        temp = []
        for j, m in enumerate(move):
          if isinstance(m, AttackTransferOrder):
            asdf = self.aaa(torch.cat([
                    x[m.src, :],
                    x[m.dst, :],
                    x1[m.src, 3:],
                    x1[m.dst, 1:],
                    torch.tensor([
                        m.armies,
                        0.6 * m.armies - 0.7 * (x1[m.dst, 3] + x1[m.dst, 4])
                        ])]))# 0.6 attackers - 0.7 defenders
            # asdf = self.bbb(F.relu(self.drop(asdf)))
            asdf = F.relu(self.drop(asdf))
          elif isinstance(m, DeployOrder):
            asdf = self.ccc(torch.cat([
                x[m.target, :], x1[m.target, 3:], torch.tensor([m.armies])
            ]))
            # asdf = self.ddd(F.relu(self.drop(asdf)))
            asdf = F.relu(self.drop(asdf))
          else:
            raise Exception(m)
          temp.append(asdf)
        temp = torch.stack(temp)

        attention = F.softmax(self.att_layer(temp), dim=0)
        p[i] += (attention * self.pi_layer(temp)).sum()

      # Use attention here too?
      V = self.lin(torch.cat([self.drop(x), x1, x2], dim=1))
      V = self.lin2(F.relu(self.drop(V)))
      V = torch.tanh(V.mean())

      return V, F.log_softmax(p, dim=0) # switch to logsoftmax?

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv, GATv2Conv

class Model7(torch.nn.Module):
    def predict_policy(self): return True

    def __init__(self):
        super().__init__()
        self.graph1 = GATv2Conv(15, 10, dropout=0.25)
        self.graph2 = GATv2Conv(10 + 15, 10, dropout=0.25)
        self.graph3 = GATv2Conv(10 + 15, 10, dropout=0.25)

        self.v_transform = Linear(10+15+4, 20)
        self.v_attention = Linear(20, 1)
        self.v_value = Linear(20, 10)
        self.v_layer = Linear(10, 1)

        self.attack_transform = Linear(20+30+2-4, 20)
        self.deploy_transform = Linear(10+15+1-3, 20)

        self.order_attention = Linear(20, 1)
        self.order_value = Linear(20, 1)

        self.drop = Dropout(0.25) # use dropout

    def forward(self, x1, x2, edges, moves):
      x2 = torch.tile(x2, (x1.size()[0], 1))

      x = self.graph1(x1, edges)
      x = F.relu(x)
      x = self.graph2(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)
      x = self.graph3(torch.cat([x, x1], dim=1), edges)
      x = F.relu(x)

      p = torch.zeros(len(moves))
      for i, move in enumerate(moves):
        order_tensors = []
        for j, m in enumerate(move):
          if isinstance(m, AttackTransferOrder):
            attack_tensor = self.attack_transform(torch.cat([
              x[m.src, :],
              x[m.dst, :],
              x1[m.src, 3:],
              x1[m.dst, 1:],
              torch.tensor([
                m.armies,
                0.6 * m.armies - 0.7 * (x1[m.dst, 3] + x1[m.dst, 4])
              ])
            ]))# 0.6 attackers - 0.7 defenders
            order_tensors.append(attack_tensor)
          elif isinstance(m, DeployOrder):
            deploy_tensor = self.deploy_transform(torch.cat([
                x[m.target, :], x1[m.target, 3:], torch.tensor([m.armies])
            ]))
            order_tensors.append(deploy_tensor)
          else:
            raise Exception(m)

        order_tensors = self.drop(F.relu(torch.stack(order_tensors)))
        attention = F.softmax(self.order_attention(order_tensors), dim=0)
        p[i] = (attention * self.order_value(order_tensors)).sum()

      V = torch.cat([x, x1, x2], dim=1)
      V = self.drop(F.relu(self.v_transform(V)))
      attention = F.softmax(self.v_attention(V), dim=0)
      V = F.relu((attention * self.v_value(V)).sum(dim=0))
      V = torch.tanh(self.v_layer(V))[0]
      return V, F.log_softmax(p, dim=0)

