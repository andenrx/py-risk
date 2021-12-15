from .orders import *
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    class StateData(Data):
      def __inc__(self, key, value, *args, **kwargs):
        if 'btch' in key:
          return self.num_moves
        elif 'index' in key or 'face' in key or 'src' in key or 'dst' in key or 'tgt' in key:
          return self.num_nodes
        else:
          return 0

      def __cat_dim__(self, key, value, *args, **kwargs):
        if 'global' in key:
          return None
        elif 'index' in key or 'face' in key:
          return 1
        else:
          return 0

except ImportError:
    pass
    
def extract_from_orders(moves, state):
  asrcs, adsts, abtch, aarmies = [], [], [], []
  tsrcs, tdsts, tbtch, tarmies = [], [], [], []
  dtgts, dbtch,        darmies = [], [], []
  for i, orders in enumerate(moves):
    for move in orders:
      if isinstance(move, AttackTransferOrder) and state.owner[move.src] != state.owner[move.dst]:
        asrcs.append(move.src)
        adsts.append(move.dst)
        abtch.append(i)
        aarmies.append(move.armies)
      elif isinstance(move, AttackTransferOrder) and state.owner[move.src] == state.owner[move.dst]:
        tsrcs.append(move.src)
        tdsts.append(move.dst)
        tbtch.append(i)
        tarmies.append(move.armies)
      elif isinstance(move, DeployOrder):
        dtgts.append(move.target)
        dbtch.append(i)
        darmies.append(move.armies)
      else:
        raise Exception(f"Move is of type '{type(move)}'")
  return {
      'asrcs':   torch.tensor(asrcs,   dtype=torch.long),
      'adsts':   torch.tensor(adsts,   dtype=torch.long),
      'abtch':   torch.tensor(abtch,   dtype=torch.long),
      'aarmies': torch.tensor(aarmies, dtype=torch.long),
      'tsrcs':   torch.tensor(tsrcs,   dtype=torch.long),
      'tdsts':   torch.tensor(tdsts,   dtype=torch.long),
      'tbtch':   torch.tensor(tbtch,   dtype=torch.long),
      'tarmies': torch.tensor(tarmies, dtype=torch.long),
      'dtgts':   torch.tensor(dtgts,   dtype=torch.long),
      'dbtch':   torch.tensor(dbtch,   dtype=torch.long),
      'darmies': torch.tensor(darmies, dtype=torch.long),
  }

def build_order_data(moves, state, x1):
  data = extract_from_orders(moves, state)
  bonus_features = torch.tensor([[
    x1[(x1[:,5+j] == 1) & (torch.arange(20) != i), 0].mean() if x1[i,5+j] == 1 else 0
    for j in range(10)] for i in range(20)
  ])
  
  data["attack_data"] = torch.cat([
    x1[data["asrcs"], :],
    x1[data["adsts"], :],
    bonus_features[data["asrcs"], :],
    bonus_features[data["adsts"], :],
    data["aarmies"].view(-1, 1),
    (0.6 * data["aarmies"] - 0.7 * (x1[data["adsts"], 3] + x1[data["adsts"], 4])).view(-1, 1)
  ], dim=1)
  data["transfer_data"] = torch.cat([
      x1[data["tsrcs"], :],
      x1[data["tdsts"], :],
      bonus_features[data["tsrcs"], :],
      bonus_features[data["tdsts"], :],
      data["tarmies"].view(-1, 1),
  ], dim=1)
  data["deploy_data"] = torch.cat([
      x1[data["dtgts"], :],
      bonus_features[data["dtgts"], :],
      data["darmies"].view(-1, 1)
  ], dim=1)
  #del data["aarmies"], data["tarmies"], data["darmies"]
  return data

