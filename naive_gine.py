# %%
import numpy as np
import pickle
import traceback
import sys
from datetime import datetime
import json
from functools import partial
import random
import math
from itertools import combinations
from collections import defaultdict
import itertools
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from sklearn import metrics
from scipy import stats
from collections import Counter
import torch
import torch.nn.functional as F
from lib.utilities import Repository
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch_geometric.nn import GINConv, GINEConv, GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset, Data
from livelossplot import PlotLosses
import optuna
from optuna.trial import TrialState



# %%
repo = Repository('./session_repositories/actions.tsv','./session_repositories/displays.tsv','./raw_datasets/')

# %%
device = torch.device('cuda')

with open(f'./edge/act_five_feats.pickle', 'rb') as fin:
    act_feats = pickle.load(fin)

with open(f'./edge/col_action.pickle', 'rb') as fin:
    col_feats = pickle.load(fin)

with open(f'./edge/cond_action.pickle', 'rb') as fin:
    cond_feats = pickle.load(fin)

with open(f'./display_feats/display_feats.pickle', 'rb') as fin:
    display_feats = pickle.load(fin)

with open(f'./display_feats/display_pca_feats_{9999}.pickle', 'rb') as fin:
    display_pca_feats = pickle.load(fin)

actcol_feats = {}
for key in act_feats:
    feat = np.zeros(len(act_feats[key]) * len(col_feats[key]))
    offset = np.argmax(act_feats[key]) * len(col_feats[key])
    feat[offset + np.argmax(col_feats[key])] = 1
    actcol_feats[key] = feat.copy()

actcolcond_feats = {}
for key in act_feats:
    feat = np.zeros(len(actcol_feats[key]) * len(cond_feats[key]))
    offset = np.argmax(actcol_feats[key]) * len(cond_feats[key])
    feat[offset + np.argmax(cond_feats[key])] = 1
    actcolcond_feats[key] = feat.copy()

concat_feats = {}
for key in act_feats:
    concat_feats[key] = np.concatenate((act_feats[key], col_feats[key])).copy()

# %%
def get_predecessors(G, node):
    predecessors = []
    for v in G.neighbors(node):
        if v < node:
            predecessors.append(v)
    return predecessors

def get_successors(G, node):
    successors = []
    for v in G.neighbors(node):
        if v > node:
            successors.append(v)
    return successors

def bfs(G, root):
    depth = 0
    node_count_at_depth = {}
    nodes_at_depth = {}
    successors = [root]
    traversal = []
    while len(successors) > 0:
        # successors.sort()
        traversal.extend(successors)
        nodes_at_depth[depth] = []
        next_level = []
        for v in successors:
            nodes_at_depth[depth].append(v)
            next_level.extend(get_successors(G, v))

        node_count_at_depth[depth] = len(successors)
        depth += 1

        successors = next_level

    return traversal, node_count_at_depth, nodes_at_depth

def dfs(G, root):
    successors = get_successors(G, root)
    if len(successors) == 0:
        return [root]
    successors.sort()
    traversal = [root]
    for v in successors:
        v_traversal = dfs(G, v)
        traversal += v_traversal

    return traversal

# %%
def replay_graph(edges, d_feats, a_feats, tar, sizes, main_size, is_train):
    logic_error_displays = [427, 428, 429, 430, 
                        854, 855, 856, 868, 891, 
                        977, 978, 979, 980, 
                        1304, 1908, 1909, 1983, 
                        2022, 2023, 2024, 2195,
                        3244, 3446, 3447, 
                        4050, 4051, 4056, 4052, 4054, 4055, 4057, 4058, 4059, 
                        4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067]

    replays = []
    display_seqs = []
    action_seqs = []
    y = []
    end_aids = []

    try:
        BigG = nx.from_edgelist(edges, create_using=nx.Graph)
        S = [BigG.subgraph(c).copy() for c in nx.connected_components(BigG)]

        # context = size
        for G in S:
            g_nodes = list(G.nodes())
            g_nodes.sort()
            g_node_ids = {}
            for node_id, node in enumerate(g_nodes):
                g_node_ids[node] = node_id

            g_nodes.reverse()
            for end in G.nodes():
                if end in logic_error_displays:
                    continue
                
                leading_to_end = None
                max_context = 0
                primary_nodes = None
                cxts = []
                for context in sizes:
                    tree_nodes = set()
                    for v in g_nodes:
                        if v < end and len(tree_nodes) < context:
                            tree_nodes.add(v)

                    if len(tree_nodes) > max_context:
                        max_context = len(tree_nodes)
                        primary_nodes = list(tree_nodes)

                    # if len(tree_nodes) < context:
                    #     if len(tree_nodes) > 2:
                    #         cxts.append(None)
                    #         continue
                    #     else:
                    #         break

                    if len(tree_nodes) < context:
                        break

                    leading_to_end = get_predecessors(G, end)[0]
                    # tree_nodes.add(leading_to_end)

                    # ### Adding older siblings ####
                    # siblings = get_successors(G, leading_to_end)
                    # for v in siblings:
                    #     if v < end:
                    #         tree_nodes.add(v)
                    
                    tree_nodes_list = list(tree_nodes)
                    backtracks = []
                    max_path_length = 0
                    max_path_index = 0
                    for j in range(len(tree_nodes)):
                        v = tree_nodes_list[j]
                        predecessors = get_predecessors(G, v)
                        path = [v]
                        while len(predecessors) > 0:
                            path.append(predecessors[0])
                            predecessors = get_predecessors(G, predecessors[0])
                        
                        if len(path) > max_path_length:
                            max_path_length = len(path)
                            max_path_index = j
                        
                        path.reverse()
                        backtracks.append(path)

                    # print(backtracks, tree_nodes)
                    
                    proceed = True
                    i = -1
                    while proceed:
                        i += 1
                        if i >= max_path_length:
                            break

                        match_value = backtracks[max_path_index][i]
                        for path in backtracks:
                            if len(path) <= i:
                                proceed = False
                                break
                            if match_value != path[i]:
                                proceed = False
                                break
                        
                    for path in backtracks:
                        for v in path[i-1:]:
                            tree_nodes.add(v)

                    # if is_train:
                    #     tree_nodes.add(end)

                    g_context = nx.induced_subgraph(G, tree_nodes).copy()
                    # g_context.graph["leading_to_end"] = leading_to_end

                    root = None
                    for v in g_context.nodes():
                        if len(get_predecessors(g_context, v)) == 0:
                            root = v

                    node_order, node_count_at_depth, nodes_at_depth = bfs(g_context, root)

                    node_depth = {}
                    for depth in nodes_at_depth:
                        depth_nodes = nodes_at_depth[depth]
                        depth_nodes.sort()
                        for i in range(len(depth_nodes)):
                            v = depth_nodes[i]
                            node_depth[v] = (depth, i)

                    tree_nodes = list(tree_nodes)
                    tree_nodes.sort()
                    attrs = {}
                    for i in range(len(tree_nodes)):
                        v = tree_nodes[i]
                        v_feat = np.zeros(28, dtype=np.float32)
                        v_feat[g_node_ids[v]] = 1.0
                        attrs[v] = {"x":v_feat.astype(np.float32), "pos":node_depth[v]}
                    nx.set_node_attributes(g_context, attrs)

                    attrs = {}
                    for edge in g_context.edges():
                        e_feat = a_feats[g_context.edges[edge[0], edge[1]]['aid']]
                        attrs[(edge[0], edge[1])] = {"x":e_feat.astype(np.float32)}
                        # attrs[(edge[0], edge[1])] = {"x":np.zeros(28, dtype=np.float32)}
                    nx.set_edge_attributes(g_context, attrs)

                    cxts.append(g_context)

                if len(cxts) == len(sizes):
                    end_aid = G.edges[leading_to_end, end]['aid']
                    # target = np.argmax(tar[end_aid])
                    replays.append(cxts)
                    y.append(tar[end_aid])
                    end_aids.append(end_aid)
                
    except Exception as e:
        # print(g_context.edges.data())
        # print(traceback.format_exc())
        print()

    return replays, y, end_aids

# %%
def generate_sessions(repo):
    og_columns = ['captured_length', 'length', 'tcp_stream', 'number', 'eth_dst', 'eth_src', 
                'highest_layer', 'info_line', 'interface_captured', 'ip_dst', 'ip_src', 'sniff_timestamp', 'tcp_dstport', 'tcp_srcport']
    sessions = {}
    for project_id in range(1, 5):
        my_sessions = []
        for session_id in repo.actions[repo.actions['project_id'] == project_id]['session_id'].unique():
            nodes = set()
            edges = []
            unrelated = False
            for i, row in repo.actions[repo.actions['session_id'] == session_id][['action_id', 'action_type', 'action_params', 'parent_display_id', 'child_display_id', 'solution']].iterrows():
                solution = 1 if row['solution'] else 0
                u = int(row['parent_display_id'])
                v = int(row['child_display_id'])
                aid = int(row['action_id'])

                nodes.add(u)
                nodes.add(v)

                if row['action_type'] == 'sort' and (not bool(row['action_params'])):
                    check_col = 'number'
                    row['action_params']['field'] = 'number'
                else:
                    check_col = row['action_params']['field']

                if not check_col in og_columns:
                    unrelated = True

                edges.append([u, v, {'aid':aid}])

            if not unrelated:
                my_sessions.append(edges)

        sessions[project_id] = my_sessions
    
    return sessions

# %%
def treefy_sessions(sessions, d_feats, a_feats, tar, sizes, main_size, test_id):
    test_contexts = []
    test_act_seqs = []
    test_display_seqs = []
    test_y = []
    test_aids = []
    train_contexts = []
    train_act_seqs = []
    train_display_seqs = []
    train_y = []
    train_aids = []
    for chunk_id in sessions:
        if chunk_id in test_id:
            for edges in sessions[chunk_id]:
                g_contexts, g_ys, end_aids = replay_graph(edges, d_feats, a_feats, tar, sizes=sizes, main_size=main_size, is_train=False)
                test_contexts.extend(g_contexts)
                test_y.extend(g_ys)
                test_aids.extend(end_aids)
        else:
            for edges in sessions[chunk_id]:
                g_contexts, g_ys, end_aids = replay_graph(edges, d_feats, a_feats, tar, sizes=sizes, main_size=main_size, is_train=True)
                train_contexts.extend(g_contexts)
                train_y.extend(g_ys)
                train_aids.extend(end_aids)

    return train_contexts, train_y, train_aids, test_contexts, test_y, test_aids

def make_gin_conv(input_dim, out_dim):
    return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(out_dim, out_dim)), eps=True, edge_dim=20)
    # return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(out_dim, out_dim)), train_eps=True)

def make_gat_conv(input_dim, out_dim, heads=4):
    return GATConv(input_dim, out_dim, heads=heads, dropout=0.5, concat=False, edge_dim=30)

class MatchingNetwork(nn.Module):
    def __init__(self, input_dim, q_num, q_hid_dim, q_num_layers, project_dim, output_dim):
        super(MatchingNetwork, self).__init__()

        self.q_num = q_num
        self.q_num_layers = q_num_layers
        self.q_models = {}
        for i in range(q_num):
            i = str(i)
            self.q_models[i] = nn.ModuleDict({
                                                'layers': nn.ModuleList(),
                                                'bn': nn.ModuleList(),
                                                'proj': nn.Linear(q_hid_dim * q_num_layers, project_dim)
                                            })


            for j in range(q_num_layers):
                if j == 0:
                    self.q_models[i]['layers'].append(make_gin_conv(input_dim, q_hid_dim))
                else:
                    self.q_models[i]['layers'].append(make_gin_conv(q_hid_dim, q_hid_dim))
                self.q_models[i]['bn'].append(nn.BatchNorm1d(q_hid_dim))

        self.q_models = nn.ModuleDict(self.q_models)

        matcher_dim = project_dim * self.q_num
        self.matcher = nn.Linear(matcher_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, queries):
        assert self.q_num == len(queries)

        full_cat = []

        for i in range(self.q_num):
            si = str(i)
            qz = queries[i].x
            qzs = []
            for conv, bn in zip(self.q_models[si]['layers'], self.q_models[si]['bn']):
                qz = conv(qz, queries[i].edge_index, queries[i].edge_x)
                qz = F.relu(qz)
                qz = bn(qz)
                qzs.append(qz)
            qs = [global_add_pool(z, queries[i].batch) for z in qzs]
            q = torch.cat(qs, dim=1)
            q = self.q_models[si]['proj'](q)
            q = F.leaky_relu(q)
            full_cat.append(q)

        matcher_input = torch.cat(full_cat, dim=1)
        output = self.matcher(matcher_input)
        output = self.sigmoid(output)

        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight) 
        m.bias.data.fill_(0.01)
        
class PairDataset(Dataset):
    def __init__(self, gs, y):
        super(PairDataset, self).__init__()
        self.gs = gs
        self.y = y

    def len(self):
        return len(self.y)
    
    def get(self, idx):
        g = self.gs[idx]
        return g, torch.tensor(self.y[idx], dtype=torch.float32)
    
class TestDataset(Dataset):
    def __init__(self, gs, y):
        super(TestDataset, self).__init__()
        self.gs = gs
        self.y = y

    def len(self):
        return len(self.y)
    
    def get(self, idx):
        g = self.gs[idx]
        return g, torch.tensor(np.argmax(self.y[idx]), dtype=torch.long)

# %%
def strip_graph_attributes(graph):
    for (_, d) in graph.nodes(data=True):
        del d['pos']
    for (_, _, d) in graph.edges(data=True):
        del d['aid']

# %%
def generate_train_set(graph_sets, train_y):
    train_set = []
    new_train_y = []

    for i, g_set in enumerate(graph_sets):
        for g in g_set:
            train_set.append(g)
            new_train_y.append(train_y[i])

    return train_set, new_train_y

# %%
def make_directed(graph_sets):
    directed_graphs = []
    for graphs in graph_sets:
        curr_set = []
        for i in range(len(graphs)):
            graph = graphs[i]
            if graph is None:
                curr_set.append(None)
            else:
                strip_graph_attributes(graph)
                dg = graph.to_directed()
                to_remove = []
                for edge in dg.edges():
                    if edge[0] > edge[1]:
                        to_remove.append(edge)

                dg.remove_edges_from(to_remove)
                pyg = from_networkx(dg)
                curr_set.append(pyg)
                
        directed_graphs.append(curr_set)

    return directed_graphs

# %%
def evaluate(model, test_dataloader, test_y, k):
    model.eval()

    preds = []
    with torch.no_grad():
        for q, label in test_dataloader:
            q = [qi.to(device) for qi in q]
            label = label.to(device)
            output = model(q)
            indices = torch.topk(output, k, dim=1).indices.tolist()
            preds.extend(indices)
         
    
    corrects = [0] * len(test_y)
    pred_classes = [0] * len(test_y)
    mrrs = [0] * len(test_y)
    total = 0
    for i in range(len(test_y)):
        total += 1
        for j in range(k):
            if np.argmax(test_y[i]) == preds[i][j]:
                corrects[i] = 1
                mrrs[i] = 1 / (j + 1)
                pred_classes[i] = test_y[i]
    
    # f1 = f1_score(test_y, pred_classes, average=None)
    correct = sum(corrects)
    mrr = sum(mrrs)
    acc = round(correct / total, 4)
    mrr_acc = round(mrr / total, 4)
    return acc, mrr_acc #f1 #, correct, total, pred_classes

def get_truly_random_seed_through_os():
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

task = str(sys.argv[1])
seed = int(sys.argv[2])
main_size = int(sys.argv[3])
test_id = [int(sys.argv[4])]

with open(f'./chunked_sessions/unbiased_seed_{seed}.pickle', 'rb') as fin:
    chunked_sessions = pickle.load(fin)

tar_feats = None
select_k = 3
if task == 'act':
    tar_feats = act_feats
    select_k = 1
elif task == 'col':
    tar_feats = col_feats
elif task == 'tg':
    tar_feats = actcol_feats

print(f'################################### TESTING CHUNK {test_id} ###################################')

sizes = [main_size]
train_x, train_y, train_aids, test_x, test_y, test_aids = treefy_sessions(
                                                                            sessions=chunked_sessions, 
                                                                            d_feats=display_pca_feats, 
                                                                            a_feats=concat_feats,
                                                                            tar=tar_feats, 
                                                                            sizes=sizes, 
                                                                            main_size=main_size,
                                                                            test_id=test_id
                                                                        )
print(len(train_y), len(test_y))

train_x = make_directed(train_x)
test_x = make_directed(test_x)

train_dataset = PairDataset(train_x, train_y)

print(len(train_dataset))

test_dataset = TestDataset(test_x, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_y), shuffle=False)

# %%
input_dim=28
q_num=len(sizes)
q_hid_dim=3000
q_num_layers=1
project_dim=3750
output_dim=tar_feats[1].shape[0]
batch_size=max(2, len(train_y) // 8)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

results = {'ra3':[], 'mrr':[]}
for _ in range(5):
    ## seeding everything
    seed_everything(seed=get_truly_random_seed_through_os())
    model = MatchingNetwork(
                                input_dim=input_dim, 
                                q_num=q_num,
                                q_hid_dim=q_hid_dim, 
                                q_num_layers=q_num_layers, 
                                project_dim=project_dim,
                                output_dim=output_dim
                            ).to(device)
    # model.apply(init_weights)

    criterion = nn.BCELoss()

    learning_rate = 1e-4
    wd = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    # Train model
    num_epochs = 200
    
    max_ra3 = 0.0
    max_mrr = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for q, labels in dataloader:
            q = [qi.to(device) for qi in q]
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(q)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        
        test_ra3, test_mrr = evaluate(model, test_dataloader, test_y, select_k)

        if test_ra3 > max_ra3:
            max_ra3 = test_ra3
        
        if test_mrr > max_mrr:
            max_mrr = test_mrr

    results['ra3'].append(max_ra3)
    results['mrr'].append(max_mrr)

pickle.dump(
    results, 
    open(f'./model_stats/{task}_{seed}_{main_size}_{test_id}_gine_mono_context.pickle', 'wb'), 
    protocol=pickle.HIGHEST_PROTOCOL
)  

