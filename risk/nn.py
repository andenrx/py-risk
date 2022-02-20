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
    def predict_policy(self): return False
    def batched(self): return False

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
    def predict_policy(self): return False
    def batched(self): return False
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
    def predict_policy(self): return False
    def batched(self): return False
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
    def predict_policy(self): return True
    def batched(self): return False
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
    def predict_policy(self): return True
    def batched(self): return False
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
    def batched(self): return False

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

def build_attack_tensor(move, x1, xc):
        ms = [m for m in move if isinstance(m, AttackTransferOrder)]
        tmp_src = [m.src for m in ms]
        tmp_dst = [m.dst for m in ms]
        attack_tensor = torch.cat([
          xc[tmp_src, :],
          xc[tmp_dst, :],
          x1[tmp_src, 3:],
          x1[tmp_dst, 1:],
          torch.tensor([[
            m.armies, 0.6 * m.armies - 0.7 * (x1[m.dst, 3] + x1[m.dst, 4])
          ] for m in ms]).view(-1, 2)
        ], dim=1)
        return attack_tensor
def build_deploy_tensor(move, x1, xc):
        ms = [m for m in move if isinstance(m, DeployOrder)]
        tmp_target = [m.target for m in ms]
        deploy_tensor = torch.cat([
          xc[tmp_target, :],
          x1[tmp_target, 3:],
          torch.tensor([[m.armies] for m in ms]).view(-1, 1)
        ], dim=1)
        return deploy_tensor

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv, GATv2Conv

class Model8(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return False
    def __init__(self):
        super().__init__()
        G_DIM = 10
        V_DIM_1 = 20
        V_DIM_2 = 10
        O_DIM_1 = 20

        self.init_layer = Linear(15, G_DIM)
        self.graph1 = GATv2Conv(1 * G_DIM + 15, G_DIM, dropout=0.25)
        self.graph2 = GATv2Conv(2 * G_DIM + 15, G_DIM, dropout=0.25)
        self.graph3 = GATv2Conv(3 * G_DIM + 15, G_DIM, dropout=0.25)

        self.v_transform = Linear(15+G_DIM+4, V_DIM_1)
        # self.v_transform2 = Linear(V_DIM_1, V_DIM_1) # delete this
        self.v_attention = Linear(V_DIM_1, 1)
        self.v_value = Linear(V_DIM_1, V_DIM_2)
        self.v_layer = Linear(V_DIM_2, 1)

        self.attack_transform = Linear(2*G_DIM+30+2-4, O_DIM_1)
        self.deploy_transform = Linear(G_DIM+15+1-3, O_DIM_1)

        self.order_attention = Linear(O_DIM_1, 1) # SWITCH BACK 10 to 1
        self.order_value = Linear(O_DIM_1, 1) # COULD TRY ADDING EXTRA LAYER AFTER THIS ????

        self.drop = Dropout(0.25) # use dropout

    def forward(self, x1, x2, edges, moves):
      x2 = torch.tile(x2, (x1.size()[0], 1))
      
      x_ = self.drop(F.relu(self.init_layer(x1)))

      xa = self.graph1(torch.cat([x_, x1], dim=1), edges)
      xa = F.relu(xa)
      xb = self.graph2(torch.cat([xa, x_, x1], dim=1), edges)
      xb = F.relu(xb)
      xc = self.graph3(torch.cat([xb, xa, x_, x1], dim=1), edges)
      xc = F.relu(xc)

      p = torch.zeros(len(moves))

      for i, move in enumerate(moves):
        attack_tensor = build_attack_tensor(move, x1, xc)
        attack_tensor = self.attack_transform(attack_tensor)
        deploy_tensor = build_deploy_tensor(move, x1, xc)
        deploy_tensor = self.deploy_transform(deploy_tensor)
        
        order_tensors = self.drop(F.relu(torch.cat([attack_tensor, deploy_tensor], dim=0)))
        attention = F.softmax(self.order_attention(order_tensors), dim=0)
        
        p[i] = (attention * self.order_value(order_tensors)).sum()

      V = torch.cat([xc, x1, x2], dim=1)
      V = self.drop(F.relu(self.v_transform(V)))
      attention = F.softmax(self.v_attention(V), dim=0)
      V = F.relu((attention * self.v_value(V)).sum(dim=0))
      V = torch.tanh(self.v_layer(V)).view(())
      return V, F.log_softmax(p, dim=0)

import torch_geometric
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import global_max_pool, GraphNorm

class Model12(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return False
    def __init__(self):
        super().__init__()
        X1_DIM = 15
        G_DIM = 20
        ORDER_UNITS = 20
        UNITS_1 = 20
        self.g1 = TransformerConv(X1_DIM, G_DIM, beta=True)
        self.g2 = TransformerConv(X1_DIM+G_DIM, G_DIM, beta=True)

        self.att = GlobalAttention(
            Sequential(Linear(X1_DIM+2*G_DIM, 1)),
            Sequential(Linear(X1_DIM+2*G_DIM, UNITS_1))
        )
        self.lin1 = Linear(UNITS_1+4, 1)

        self.attack_transform   = Linear(2*X1_DIM + 2*2*G_DIM + 2 + 2*10, ORDER_UNITS)
        self.transfer_transform = Linear(2*X1_DIM + 2*2*G_DIM + 1 + 2*10, ORDER_UNITS)
        self.deploy_transform   = Linear(  X1_DIM +   2*G_DIM + 1 + 10,   ORDER_UNITS)

        self.order_accumulate = Linear(ORDER_UNITS, 1)

        self.drop = Dropout(0.20)
        self.norm1 = GraphNorm(G_DIM)
        self.norm2 = GraphNorm(G_DIM)
        self.norm3 = GraphNorm(UNITS_1)
        self.norm4 = GraphNorm(ORDER_UNITS)

    def forward(self, x1, x2, edges, moves):
        edges = torch_geometric.utils.to_undirected(edges)

        xa = F.relu(self.g1(x1, edges))
        xa = self.norm1(xa)
        xb = F.relu(self.g2(torch.cat([x1, xa], dim=1), edges))
        xb = self.norm2(xb)
        x = self.att(torch.cat([x1, xa, xb], dim=1), torch.zeros(20, dtype=torch.long)).view(-1)
        x = self.norm3(x)
        x = self.lin1(torch.cat([F.relu(x), x2]))
        V = torch.tanh(x).view(())

        x_cat = torch.cat([xa, xb], dim=1)

        # percent of bonus owned (ignoring given terr)
        tmp = torch.tensor([[
          x1[(x1[:,5+j] == 1) & (torch.arange(20) != i), 0].mean() if x1[i,5+j] == 1 else 0
          for j in range(10)] for i in range(20)
        ])

        p = torch.zeros(len(moves))
        for i, move in enumerate(moves):
            attack_tensor   = self.build_attack_tensor(move, x1, x_cat, tmp)
            attack_tensor   = self.attack_transform(attack_tensor)
            transfer_tensor = self.build_transfer_tensor(move, x1, x_cat, tmp)
            transfer_tensor = self.transfer_transform(transfer_tensor)
            deploy_tensor   = self.build_deploy_tensor(move, x1, x_cat, tmp)
            deploy_tensor   = self.deploy_transform(deploy_tensor)

            order_tensors = torch.cat([attack_tensor, transfer_tensor, deploy_tensor], dim=0)
            order_tensors = self.norm4(order_tensors)
            order_tensors = self.drop(order_tensors)
            order_tensors = self.order_accumulate(order_tensors)

            p[i] += order_tensors.sum()

        return V, F.log_softmax(p, dim=0)

    def build_attack_tensor(self, move, x1, X, tmp1):
        ms = [m for m in move if isinstance(m, AttackTransferOrder) and x1[m.dst, 0] != 1]
        srcs = [m.src for m in ms]
        dsts = [m.dst for m in ms]

        attack_tensor = torch.cat([
            x1[srcs, :],
            x1[dsts, :],
            tmp1[srcs, :],
            tmp1[dsts, :],
            X[srcs, :],
            X[dsts, :],
            torch.tensor([[m.armies, 0.6 * m.armies - 0.7 * (x1[m.dst, 3] + x1[m.dst, 4])] for m in ms]).view(-1, 2)
        ], dim=1)
        return attack_tensor

    def build_transfer_tensor(self, move, x1, X, tmp1):
        ms = [m for m in move if isinstance(m, AttackTransferOrder) and x1[m.dst, 0] == 1]
        srcs = [m.src for m in ms]
        dsts = [m.dst for m in ms]

        transfer_tensor = torch.cat([
            x1[srcs, :],
            x1[dsts, :],
            tmp1[srcs, :],
            tmp1[dsts, :],
            X[srcs, :],
            X[dsts, :],
            torch.tensor([[m.armies] for m in ms]).view(-1, 1)
        ], dim=1)
        return transfer_tensor

    def build_deploy_tensor(self, move, x1, X, tmp1):
        ms = [m for m in move if isinstance(m, DeployOrder)]
        tgts = [m.target for m in ms]
        deploy_tensor = torch.cat([
            x1[tgts, :],
            tmp1[tgts, :],
            X[tgts, :],
            torch.tensor([[m.armies] for m in ms]).view(-1, 1)
        ], dim=1)
        return deploy_tensor

import torch_geometric
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import global_max_pool, GraphNorm, global_add_pool

class Model13(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return True
    def __init__(self, config):
        super().__init__()
        X1_DIM = 15
        G_DIM = config.G_DIM
        self.g1 = TransformerConv(X1_DIM, G_DIM, dropout=config.DROPOUT, beta=True)
        self.g2 = TransformerConv(X1_DIM+G_DIM, G_DIM, dropout=config.DROPOUT, beta=True)

        self.att = GlobalAttention(
            Sequential(Linear(X1_DIM+2*G_DIM, 1)),
            Sequential(Linear(X1_DIM+2*G_DIM, config.UNITS_1))
        )
        self.lin1 = Linear(config.UNITS_1+4, 1)

        self.attack_transform   = Linear(2*X1_DIM + 2*2*G_DIM + 2 + 2*10, config.ORDER_UNITS)
        self.transfer_transform = Linear(2*X1_DIM + 2*2*G_DIM + 1 + 2*10, config.ORDER_UNITS)
        self.deploy_transform   = Linear(  X1_DIM +   2*G_DIM + 1 + 10,   config.ORDER_UNITS)

        self.order_accumulate = Linear(config.ORDER_UNITS, config.FINAL_ORDER_UNITS)
        self.final_order_layer = Linear(config.FINAL_ORDER_UNITS, 1)

        self.drop = Dropout(config.DROPOUT)
        self.drop2 = Dropout(config.DROPOUT_2)
        self.norm1 = GraphNorm(G_DIM)
        self.norm2 = GraphNorm(G_DIM)

    def forward(self, data):
        edges = data.edge_index
        assert torch_geometric.utils.is_undirected(edges)
        x1 = data.graph_data
        x2 = data.global_data

        xa = F.relu(self.g1(x1, edges))
        xa = self.norm1(xa, data.batch)
        xb = F.relu(self.g2(torch.cat([x1, xa], dim=1), edges))
        xb = self.norm2(xb, data.batch)
        x = self.att(torch.cat([x1, xa, xb], dim=1), data.batch)
        x = self.lin1(torch.cat([F.relu(x), x2], dim=1))
        V = torch.tanh(x).view(-1)

        x_cat = torch.cat([xa, xb], dim=1)

        attack_tensor = torch.cat([data.attack_data, x_cat[data.asrcs,:], x_cat[data.adsts,:]], dim=1)
        attack_tensor = self.drop(attack_tensor)
        attack_tensor = self.attack_transform(attack_tensor)
        transfer_tensor = torch.cat([data.transfer_data, x_cat[data.tsrcs, :], x_cat[data.tdsts, :]], dim=1)
        transfer_tensor = self.drop(transfer_tensor)
        transfer_tensor = self.transfer_transform(transfer_tensor)
        deploy_tensor = torch.cat([data.deploy_data, x_cat[data.dtgts,:]], dim=1)
        deploy_tensor = self.drop(deploy_tensor)
        deploy_tensor = self.deploy_transform(deploy_tensor)

        order_tensors = torch.cat([attack_tensor, transfer_tensor, deploy_tensor], dim=0)
        order_tensors = self.drop2(order_tensors)
        order_tensors = self.order_accumulate(order_tensors)
        tmp = global_add_pool(order_tensors, batch=torch.cat([data.abtch, data.tbtch, data.dbtch], dim=0))
        assert (data.num_moves == data.num_moves[0]).all()
        tmp = self.final_order_layer(F.relu(tmp))
        tmp = tmp.reshape((-1, data.num_moves[0]))

        p = tmp

        return V, F.log_softmax(p, dim=-1)

import torch_geometric
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import SAGEConv, ResGatedGraphConv
from torch_geometric.nn import global_max_pool, GraphNorm, global_add_pool

class Model14(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return True
    def __init__(self):
        super().__init__()
        X1_DIM = 15
        ATT_UNITS = 50
        G_DIM = 50
        UNITS_1 = 50
        UNITS_2 = 50
        ORDER_UNITS = 20
        FINAL_ORDER_UNITS = 20

        self.g1 = TransformerConv(X1_DIM, G_DIM, beta=True)
        self.g2 = TransformerConv(X1_DIM+G_DIM, G_DIM, beta=True)
        self.g3 = TransformerConv(X1_DIM+2*G_DIM, G_DIM, beta=True)

        self.att = GlobalAttention(
            Sequential(Linear(X1_DIM+3*G_DIM, ATT_UNITS), LeakyReLU(), Linear(ATT_UNITS, 1)),
            Sequential(Linear(X1_DIM+3*G_DIM, ATT_UNITS), LeakyReLU(), Linear(ATT_UNITS, UNITS_1))
        )
        self.lin1 = Linear(UNITS_1+4, UNITS_2)
        self.lin2 = Linear(UNITS_2, 1)

        self.attack_transform   = Linear(2*X1_DIM + 2*3*G_DIM + 2 + 2*10, ORDER_UNITS)
        self.transfer_transform = Linear(2*X1_DIM + 2*3*G_DIM + 1 + 2*10, ORDER_UNITS)
        self.deploy_transform   = Linear(  X1_DIM +   3*G_DIM + 1 + 10,   ORDER_UNITS)

        self.attack_transform2   = Linear(ORDER_UNITS, ORDER_UNITS)
        self.transfer_transform2 = Linear(ORDER_UNITS, ORDER_UNITS)
        self.deploy_transform2   = Linear(ORDER_UNITS, ORDER_UNITS)


        self.order_accumulate = Linear(ORDER_UNITS, FINAL_ORDER_UNITS)
        self.final_order_layer = Linear(FINAL_ORDER_UNITS, 1)

        self.norm1 = GraphNorm(G_DIM)
        self.norm2 = GraphNorm(G_DIM)
        self.norm3 = GraphNorm(G_DIM)

    def forward(self, data):
        edges = data.edge_index
        assert torch_geometric.utils.is_undirected(edges)
        x1 = data.graph_data
        x2 = data.global_data

        xa = F.relu(self.g1(x1, edges))
        xa = self.norm1(xa, data.batch)
        xb = F.relu(self.g2(torch.cat([x1, xa], dim=1), edges))
        xb = self.norm2(xb, data.batch)
        xc = F.relu(self.g3(torch.cat([x1, xa, xb], dim=1), edges))
        xc = self.norm3(xc, data.batch)
        x = self.att(torch.cat([x1, xa, xb, xc], dim=1), data.batch)
        x = self.lin1(torch.cat([F.relu(x), x2], dim=1))
        x = F.relu(x)
        x = self.lin2(x)
        V = torch.tanh(x).view(-1)

        x_cat = torch.cat([xa, xb, xc], dim=1)

        attack_tensor = torch.cat([data.attack_data, x_cat[data.asrcs,:], x_cat[data.adsts,:]], dim=1)
        attack_tensor = self.attack_transform(attack_tensor)
        attack_tensor = F.relu(attack_tensor)
        attack_tensor = self.attack_transform2(attack_tensor)
        transfer_tensor = torch.cat([data.transfer_data, x_cat[data.tsrcs, :], x_cat[data.tdsts, :]], dim=1)
        transfer_tensor = self.transfer_transform(transfer_tensor)
        transfer_tensor = F.relu(transfer_tensor)
        transfer_tensor = self.transfer_transform2(transfer_tensor)
        deploy_tensor = torch.cat([data.deploy_data, x_cat[data.dtgts,:]], dim=1)
        deploy_tensor = self.deploy_transform(deploy_tensor)
        deploy_tensor = F.relu(deploy_tensor)
        deploy_tensor = self.deploy_transform2(deploy_tensor)

        order_tensors = torch.cat([attack_tensor, transfer_tensor, deploy_tensor], dim=0)
        order_tensors = self.order_accumulate(order_tensors)
        tmp = global_add_pool(order_tensors, batch=torch.cat([data.abtch, data.tbtch, data.dbtch], dim=0))
        assert (data.num_moves == data.num_moves[0]).all()
        tmp = self.final_order_layer(F.relu(tmp))
        tmp = tmp.reshape((-1, data.num_moves[0]))

        p = tmp

        return V, F.log_softmax(p, dim=-1)


class Model14v2(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return True
    def __init__(self):
        super().__init__()
        X1_DIM = 15
        G_DIM = 50
        DROPOUT = 0.1
        DROPOUT_2 = 0.3
        ATT_UNITS = 50
        UNITS_1 = UNITS_2 = 10
        ORDER_UNITS = 20
        FINAL_ORDER_UNITS = 20
        self.g1 = TransformerConv(X1_DIM, G_DIM, dropout=DROPOUT, beta=True)
        self.g2 = TransformerConv(X1_DIM+G_DIM, G_DIM, dropout=DROPOUT, beta=True)
        self.g3 = TransformerConv(X1_DIM+2*G_DIM, G_DIM, dropout=DROPOUT, beta=True)

        self.att = GlobalAttention(
            Sequential(Linear(X1_DIM+3*G_DIM, ATT_UNITS), LeakyReLU(), Linear(ATT_UNITS, 1)),
            Sequential(Linear(X1_DIM+3*G_DIM, ATT_UNITS), LeakyReLU(), Linear(ATT_UNITS, UNITS_1))
        )
        self.lin1 = Linear(UNITS_1+4, UNITS_2)
        self.lin2 = Linear(UNITS_2, 1)

        self.attack_transform   = Linear(2*X1_DIM + 2*3*G_DIM + 2 + 2*10, ORDER_UNITS)
        self.transfer_transform = Linear(2*X1_DIM + 2*3*G_DIM + 1 + 2*10, ORDER_UNITS)
        self.deploy_transform   = Linear(  X1_DIM +   3*G_DIM + 1 + 10,   ORDER_UNITS)

        self.attack_transform2   = Linear(ORDER_UNITS, ORDER_UNITS)
        self.transfer_transform2 = Linear(ORDER_UNITS, ORDER_UNITS)
        self.deploy_transform2   = Linear(ORDER_UNITS, ORDER_UNITS)


        self.order_accumulate = Linear(ORDER_UNITS, FINAL_ORDER_UNITS)
        self.final_order_layer = Linear(FINAL_ORDER_UNITS, 1)

        self.drop2 = Dropout(DROPOUT_2)
        self.norm1 = GraphNorm(G_DIM)
        self.norm2 = GraphNorm(G_DIM)
        self.norm3 = GraphNorm(G_DIM)

    def forward(self, data):
        edges = data.edge_index
        assert torch_geometric.utils.is_undirected(edges)
        x1 = data.graph_data
        x2 = data.global_data

        xa = F.relu(self.g1(x1, edges))
        xa = self.norm1(xa, data.batch)
        xb = F.relu(self.g2(torch.cat([x1, xa], dim=1), edges))
        xb = self.norm2(xb, data.batch)
        xc = F.relu(self.g3(torch.cat([x1, xa, xb], dim=1), edges))
        xc = self.norm3(xc, data.batch)
        x = self.att(torch.cat([x1, xa, xb, xc], dim=1), data.batch)
        x = self.lin1(torch.cat([F.relu(x), x2], dim=1))
        x = F.relu(x)
        x = self.drop2(x)
        x = self.lin2(x)
        V = torch.tanh(x).view(-1)

        x_cat = torch.cat([xa, xb, xc], dim=1)

        attack_tensor = torch.cat([data.attack_data, x_cat[data.asrcs,:], x_cat[data.adsts,:]], dim=1)
        attack_tensor = self.drop2(attack_tensor)
        attack_tensor = self.attack_transform(attack_tensor)
        attack_tensor = F.relu(attack_tensor)
        attack_tensor = self.attack_transform2(attack_tensor)
        transfer_tensor = torch.cat([data.transfer_data, x_cat[data.tsrcs, :], x_cat[data.tdsts, :]], dim=1)
        transfer_tensor = self.drop2(transfer_tensor)
        transfer_tensor = self.transfer_transform(transfer_tensor)
        transfer_tensor = F.relu(transfer_tensor)
        transfer_tensor = self.transfer_transform2(transfer_tensor)
        deploy_tensor = torch.cat([data.deploy_data, x_cat[data.dtgts,:]], dim=1)
        deploy_tensor = self.drop2(deploy_tensor)
        deploy_tensor = self.deploy_transform(deploy_tensor)
        deploy_tensor = F.relu(deploy_tensor)
        deploy_tensor = self.deploy_transform2(deploy_tensor)

        order_tensors = torch.cat([attack_tensor, transfer_tensor, deploy_tensor], dim=0)
        order_tensors = self.drop2(order_tensors)
        order_tensors = self.order_accumulate(order_tensors)
        tmp = global_add_pool(order_tensors, batch=torch.cat([data.abtch, data.tbtch, data.dbtch], dim=0))
        assert (data.num_moves == data.num_moves[0]).all()
        tmp = self.final_order_layer(F.relu(tmp))
        tmp = tmp.reshape((-1, data.num_moves[0]))

        p = tmp

        return V, F.log_softmax(p, dim=-1)


class Model16(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return True
    def __init__(self, config):
        super().__init__()

        self.lin1 = Linear(5*20 + 5, config.UNITS_1)
        self.lin2 = Linear(config.UNITS_1, config.UNITS_2)
        self.lin3 = Linear(config.UNITS_2, config.UNITS_3*20)
        self.lin4 = Linear(config.UNITS_3*20, 1)

        self.drop = Dropout(config.DROPOUT)
        self.UNITS = config.UNITS_3

        self.attack_transform = Linear(2*self.UNITS+1+10, config.UNITS_4)
        self.transfer_transform = Linear(2*self.UNITS+1+10, config.UNITS_4)
        self.deploy_transform = Linear(self.UNITS+1+5, config.UNITS_4)
        self.attack_transform2 = Linear(config.UNITS_4, config.UNITS_4)
        self.transfer_transform2 = Linear(config.UNITS_4, config.UNITS_4)
        self.deploy_transform2 = Linear(config.UNITS_4, config.UNITS_4)

        self.order_layer_2 = Linear(config.UNITS_4, config.UNITS_5)

        self.final_order_layer = Linear(config.UNITS_5, 1)


    def forward(self, data):
      x = data.graph_features
      x = x.reshape(-1, 5*20)
      x = torch.cat([x, data.income, data.total_armies], dim=1)
      x = self.lin1(x)
      x = F.relu(x)
      x = self.drop(x)
      x = self.lin2(x)
      x = F.relu(x)
      x = self.drop(x)
      x = self.lin3(x)
      x = F.relu(x)
      x = self.drop(x)

      x_cat = x.reshape(-1, self.UNITS)
      x_cat = torch.cat([x_cat, data.graph_features], dim=1)

      attack_tensor = torch.cat([data.aarmies.view(-1, 1), x_cat[data.asrcs,:], x_cat[data.adsts,:]], dim=1)
      attack_tensor = self.attack_transform(attack_tensor)
      attack_tensor = F.relu(attack_tensor)
      attack_tensor = self.attack_transform2(attack_tensor)
      transfer_tensor = torch.cat([data.tarmies.view(-1, 1), x_cat[data.tsrcs, :], x_cat[data.tdsts, :]], dim=1)
      transfer_tensor = self.transfer_transform(transfer_tensor)
      transfer_tensor = F.relu(transfer_tensor)
      transfer_tensor = self.transfer_transform2(transfer_tensor)
      deploy_tensor = torch.cat([data.darmies.view(-1, 1), x_cat[data.dtgts,:]], dim=1)
      deploy_tensor = self.deploy_transform(deploy_tensor)
      deploy_tensor = F.relu(deploy_tensor)
      deploy_tensor = self.deploy_transform2(deploy_tensor)

      order_tensors = torch.cat([attack_tensor, transfer_tensor, deploy_tensor], dim=0)
      order_tensors = F.relu(order_tensors)

      order_tensors = self.order_layer_2(order_tensors)

      tmp = global_add_pool(order_tensors, batch=torch.cat([data.abtch, data.tbtch, data.dbtch], dim=0))
      assert (data.num_moves == data.num_moves[0]).all()
      tmp = self.final_order_layer(F.relu(tmp))
      tmp = tmp.reshape((-1, data.num_moves[0]))
      p = tmp

      pi = F.log_softmax(p, dim=-1)


      x = self.lin4(x)

      x = torch.tanh(x).view(-1)

      return x, pi

import torch_geometric
import torch_sparse
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import SAGEConv, ResGatedGraphConv, GraphNorm, BatchNorm
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn import JumpingKnowledge
import torch_geometric
import torch_sparse
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import SAGEConv, ResGatedGraphConv, GraphNorm, BatchNorm
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn import JumpingKnowledge

class Model15(torch.nn.Module):
    def predict_policy(self): return True
    def batched(self): return True

    def __init__(self):
        super().__init__()
        self.init = Linear(5, 20)

        self.b1 = TransformerConv(20, 20, dropout=0.50, beta=True)
        self.g1 = TransformerConv(20 + 20, 20, dropout=0.50, beta=True)
        self.b2 = TransformerConv(20, 20, dropout=0.50, beta=True)
        self.g2 = TransformerConv(20 + 20, 20, dropout=0.50, beta=True)
        self.b3 = TransformerConv(20, 20, dropout=0.50, beta=True)
        self.g3 = TransformerConv(20 + 20, 60, dropout=0.50, beta=True)

        self.final1 = Linear(60+5, 60)
        self.final2 = Linear(60, 1)
        self.drop = Dropout(0.50)

    def forward(self, data):
      x = data.graph_features

      # Init layer
      x = self.init(x)
      x1 = x = F.relu(x)

      # Bonus layer 1
      b = x[data.bonus_nodes]
      b = F.relu(self.b1(b, data.bonus_edges))
      b = global_add_pool(b, data.bonus_batch)
      x2 = b = torch_sparse.spmm(data.bonus_mapping, data.bonus_values_normed[data.bonus_mapping[1]], data.num_nodes, data.num_bonuses.sum(), b)

      # Graph layer 1
      x = torch.cat([x, b], dim=1)
      x3 = x = F.relu(self.g1(x, data.graph_edges))

      # Bonus layer 2
      b = x[data.bonus_nodes]
      b = F.relu(self.b2(b, data.bonus_edges))
      b = global_add_pool(b, data.bonus_batch)
      x4 = b = torch_sparse.spmm(data.bonus_mapping, data.bonus_values_normed[data.bonus_mapping[1]], data.num_nodes, data.num_bonuses.sum(), b)

      # Graph layer 2
      x = torch.cat([x, b], dim=1)
      x5 = x = F.relu(self.g2(x, data.graph_edges))

      # Bonus layer 3
      b = x[data.bonus_nodes]
      b = F.relu(self.b3(b, data.bonus_edges))
      b = global_add_pool(b, data.bonus_batch)
      x6 = b = torch_sparse.spmm(data.bonus_mapping, data.bonus_values_normed[data.bonus_mapping[1]], data.num_nodes, data.num_bonuses.sum(), b)

      # Graph layer 3
      x = torch.cat([x, b], dim=1)
      x7 = x = F.relu(self.g3(x, data.graph_edges))

      x = global_mean_pool(x, data.batch)

      # assert x.size(0) == data.income.size(0) == data.total_armies.size(0)
      x = torch.cat([x, data.income, data.total_armies], dim=1)
      x = self.final1(x)
      x = F.relu(x)
      x = self.drop(x)
      x = self.final2(x)

      x = torch.tanh(x).view(-1)
      pi = torch.log_softmax(torch.zeros((len(data.num_moves), data.num_moves[0])), dim=1)

      return x, pi

import torch_geometric
import torch_sparse
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Identity, MultiheadAttention, Sequential
from torch.nn import ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, Linear, GatedGraphConv, GATConv
from torch_geometric.nn import GATv2Conv, TransformerConv, GlobalAttention
from torch_geometric.nn import SAGEConv, ResGatedGraphConv, GraphNorm, BatchNorm
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn import JumpingKnowledge

class Model18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        UNITS = 30
        DROPOUT = 0.25
        self.init = Linear(5+2, UNITS)
        self.g1 = TransformerConv(UNITS, UNITS, dropout=DROPOUT, beta=True)
        self.g2 = TransformerConv(UNITS, UNITS, dropout=DROPOUT, beta=True)
        self.g3 = TransformerConv(2*UNITS, UNITS, dropout=DROPOUT, beta=True)
        self.b1 = TransformerConv(UNITS, UNITS, dropout=DROPOUT, beta=True)
        self.final1 = Linear(UNITS, UNITS)
        self.final2 = Linear(UNITS, 1)

    def forward(self, data):
      # income_repeated = torch.repeat_interleave(data.income, data.num_nodes, dim=0)
      income_repeated = data.income[data.batch]
      x = torch.cat([data.graph_features, income_repeated], dim=1)
      x = F.relu(self.init(x))
      x = F.relu(self.g1(x, data.graph_edges))
      x = F.relu(self.g2(x, data.graph_edges))
      b = x[data.bonus_nodes]
      b = F.relu(self.b1(b, data.bonus_edges))
      b = global_add_pool(b, data.bonus_batch)
      b = torch_sparse.spmm(data.bonus_mapping, data.bonus_values_normed[data.bonus_mapping[1]], data.num_nodes, data.num_bonuses.sum(), b)
      x = torch.cat([x, b], dim=1)
      x = F.relu(self.g3(x, data.graph_edges))
      x = F.relu(self.final1(x))
      v = global_mean_pool(x, data.batch)
      v = self.final2(v).view(-1)
      return torch.tanh(v)

    def prep(self, state, state_value=None, move=None, move_value=None, visits=None, p1=1, p2=2):
      assert state.winner() is not None or (state.owner == p1).any() and (state.owner == p2).any()
      edges = state.mapstruct.edgeTensor()
      mask, nodes, values, b_edges = bonus_tensors(state.mapstruct)
      z = torch.zeros(values.size(), dtype=torch.long)
      z.index_add_(0, mask, torch.ones(mask.size(), dtype=torch.long))
      assert torch_geometric.utils.is_undirected(edges)
      assert torch_geometric.utils.is_undirected(b_edges)
      i1, i2 = state.income(p1), state.income(p2)
      graph_features = to_tensor(state, p1, p2)
      return StateData(
        map=int(state.mapstruct.id),
        graph_features=graph_features,
        graph_edges=edges,
        bonus_edges=b_edges,
        bonus_batch=mask,
        bonus_nodes=nodes,
        bonus_values=values,
        bonus_values_normed=values / z,
        bonus_mapping=get_bonus_mapping(state.mapstruct),
        income=torch.tensor([i1, i2]).view(1,-1),
        total_armies=graph_features[:,2:].sum(dim=0).view(1,-1),
        num_nodes=len(graph_features),
        num_bonuses=len(values),
        state=state,
        state_value=state_value,
        move=move,
        move_value=move_value,
        visits=visits,
        #move_embedding=move_embedding(mapstruct, move),
      )
from functools import lru_cache
from itertools import product

@lru_cache(32)
def get_mapstruct(id):
  gameid = api.createGame([1,2], botgame=True, mapid=id)
  return api.getMapStructure(gameid, botgame=True)

@lru_cache(32)
def get_bonus_mapping(mapstruct):
  a = []
  for i, bonus in enumerate(mapstruct.bonuses):
    for j in bonus.terr:
      a.append([j, i])
  a = torch.tensor(a).T
  return a

def to_tensor(self, p1, p2):
    graph_features = torch.tensor(np.array([
        self.owner == p1,
        self.owner == p2,
        self.armies * (self.owner == p1),
        self.armies * (self.owner == p2),
        self.armies * (self.owner == 0),
    ]), dtype=torch.float).T
    return graph_features

@lru_cache(32)
def bonus_tensors(mapstruct):
  mask = []
  nodes = []
  values = []
  edges = []

  for i, bonus in enumerate(mapstruct.bonuses):
    edges += list(product(range(len(nodes), len(nodes) + len(bonus.terr)), range(len(nodes), len(nodes) + len(bonus.terr))))
    nodes += list(bonus.terr)
    mask += [i] * len(bonus.terr)
    values.append(bonus.value)
  return torch.tensor(mask, dtype=torch.long), torch.tensor(nodes, dtype=torch.long), torch.tensor(values, dtype=torch.float), torch.tensor(edges, dtype=torch.long).T

def first(x):
  return next(iter(x))
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class StateData(Data):
  def __inc__(self, key, value, *args, **kwargs):
    if key == 'graph_edges':
      return self.num_nodes
    elif key == 'bonus_edges':
      return len(self.bonus_nodes)
    elif key == 'bonus_batch':
      return self.num_bonuses
    elif key == 'bonus_nodes':
      return self.num_nodes
    elif key == 'bonus_mapping':
      return torch.tensor([[self.num_nodes], [self.num_bonuses]])
    elif "btch" in key:
      return self.num_moves
    elif "src" in key or "dst" in key or "tgt" in key:
      return self.num_nodes
    else:
      return 0

  def __cat_dim__(self, key, value, *args, **kwargs):
    if 'edges' in key or 'mapping' in key:# or key == 'move_embedding':
      return 1
    else:
      return 0


