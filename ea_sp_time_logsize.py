# %%
import sys
import numpy as np
import pickle
import traceback
import time
import random
import statistics
import os
import networkx as nx
import torch
import torch.nn.functional as F
from lib.utilities import Repository
from torch import nn
from torch_geometric.nn import GINEConv, GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Dataset

data_path = '/data/gpfs/projects/punim2258'

# %%
repo = Repository('./session_repositories/actions.tsv','./session_repositories/displays.tsv','./raw_datasets/')

with open(f'{data_path}/network_data/edge/act_five_feats.pickle', 'rb') as fin:
    act_feats = pickle.load(fin)

with open(f'{data_path}/network_data/edge/col_action.pickle', 'rb') as fin:
    col_feats = pickle.load(fin)

with open(f'{data_path}/network_data/edge/cond_action.pickle', 'rb') as fin:
    cond_feats = pickle.load(fin)

with open(f'{data_path}/network_data/display_feats/display_feats.pickle', 'rb') as fin:
    display_feats = pickle.load(fin)

with open(f'{data_path}/network_data/display_feats/display_pca_feats_{9999}.pickle', 'rb') as fin:
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
def replay_graph(edges, d_feats, a_feats, tar, min_size, main_size, is_train):
    logic_error_displays = [427, 428, 429, 430, 
                        854, 855, 856, 868, 891, 
                        977, 978, 979, 980, 
                        1304, 1908, 1909, 1983, 
                        2022, 2023, 2024, 2195,
                        3244, 3446, 3447, 
                        4050, 4051, 4056, 4052, 4054, 4055, 4057, 4058, 4059, 
                        4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067]

    replays = []
    seqs = []
    y = []
    end_aids = []

    try:
        BigG = nx.from_edgelist(edges, create_using=nx.Graph)
        S = [BigG.subgraph(c).copy() for c in nx.connected_components(BigG)]

        # context = size
        for G in S:
            g_nodes = list(G.nodes())
            g_nodes.sort()
            g_nodes.reverse()
            for end in G.nodes():
                if end in logic_error_displays:
                    continue
                
                leading_to_end = None

                tree_nodes = set()
                sorted_nodes = []
                for v in g_nodes:
                    if v < end and len(tree_nodes) < main_size:
                        tree_nodes.add(v)

                    if v < end:
                        sorted_nodes.append(v)

                if len(tree_nodes) < min_size:
                    continue

                reported_size = len(tree_nodes)
                
                seq = []
                sorted_nodes.sort()
                for v in sorted_nodes[min_size:]:
                    u = get_predecessors(G, v)[0]
                    v_aid = G.edges[u, v]['aid']
                    seq.append(v_aid)

                leading_to_end = get_predecessors(G, end)[0]

                # me = end
                # my_predecessors = get_predecessors(G, me)
                # seq = []
                # while len(my_predecessors) > 0:
                #     v_aid = G.edges[my_predecessors[0], me]['aid']
                #     seq.append(v_aid)
                #     me = my_predecessors[0]
                #     my_predecessors = get_predecessors(G, me)
                # seq.pop()
                # seq.reverse()

                
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

                g_context = nx.induced_subgraph(G, tree_nodes).copy()

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
                    v_feat = d_feats[v]
                    attrs[v] = {"x":v_feat.astype(np.float32), "pos":node_depth[v]}
                nx.set_node_attributes(g_context, attrs)

                attrs = {}
                for edge in g_context.edges():
                    e_feat = a_feats[g_context.edges[edge[0], edge[1]]['aid']]
                    attrs[(edge[0], edge[1])] = {"x":e_feat.astype(np.float32)}
                nx.set_edge_attributes(g_context, attrs)

                end_aid = G.edges[leading_to_end, end]['aid']
                
                replays.append(g_context)
                end_aids.append(end_aid)

                if reported_size == main_size:
                    seq.append(end_aid)
                
                if len(seq) > 0 and reported_size == main_size:
                    seqs.append(seq)

                if is_train:
                    y.append(tar[end_aid])
                else:
                    target = np.argmax(tar[end_aid])
                    y.append(target)

                
    except Exception as e:
        # print(g_context.edges.data())
        print(traceback.format_exc())

    return replays, seqs, y, end_aids

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
def treefy_sessions(sessions, d_feats, a_feats, tar, min_size, main_size, test_id, train_id):
    test_contexts = []
    test_seqs = []
    test_y = []
    test_aids = []
    train_contexts = []
    train_seqs = []
    train_y = []
    train_aids = []
    for chunk_id in sessions:
        if chunk_id in test_id:
            for edges in sessions[chunk_id]:
                g_contexts, seqs, g_ys, end_aids = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, main_size=main_size, is_train=False)
                test_contexts.extend(g_contexts)
                test_seqs.extend(seqs)
                test_y.extend(g_ys)
                test_aids.extend(end_aids)
        elif chunk_id in train_id:
            for edges in sessions[chunk_id]:
                g_contexts, seqs, g_ys, end_aids = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, main_size=main_size, is_train=True)
                train_contexts.extend(g_contexts)
                train_seqs.extend(seqs)
                train_y.extend(g_ys)
                train_aids.extend(end_aids)

    return train_contexts, train_seqs, train_y, train_aids, test_contexts, test_seqs, test_y, test_aids



def make_gin_conv(input_dim, out_dim):
    return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(out_dim, out_dim)), eps=True, edge_dim=20)
    # return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim)), eps=True, edge_dim=20)

def make_gat_conv(input_dim, out_dim, heads=4):
    return GATConv(input_dim, out_dim, heads=heads, dropout=0.5, concat=False, edge_dim=30)

class MatchingNetwork(nn.Module):
    def __init__(self, input_dim, q_hid_dim, q_num_layers, project_dim, output_dim, lstm_hid_dim, lstm_num_layers, lstm_project_dim, device):
        super(MatchingNetwork, self).__init__()

        self.device = device

        self.g_layers = nn.ModuleList()
        self.g_batch_norms = nn.ModuleList()
        self.g_num_layers = q_num_layers
        self.g_projector = nn.Linear(q_hid_dim * q_num_layers, project_dim)

        for i in range(q_num_layers):
            if i == 0:
                self.g_layers.append(make_gin_conv(input_dim, q_hid_dim))
            else:
                self.g_layers.append(make_gin_conv(q_hid_dim, q_hid_dim))
            self.g_batch_norms.append(nn.BatchNorm1d(q_hid_dim))

        self.q_layers = nn.ModuleList()
        self.q_batch_norms = nn.ModuleList()
        self.q_num_layers = q_num_layers
        self.q_projector = nn.Linear(q_hid_dim * q_num_layers, project_dim)

        for i in range(q_num_layers):
            if i == 0:
                self.q_layers.append(make_gin_conv(input_dim, q_hid_dim))
            else:
                self.q_layers.append(make_gin_conv(q_hid_dim, q_hid_dim))
            self.q_batch_norms.append(nn.BatchNorm1d(q_hid_dim))

        self.lstm_hid_dim = lstm_hid_dim
        self.lstm_num_layers = lstm_num_layers
        # self.lstm = nn.LSTM(project_dim * 4, lstm_hid_dim, lstm_num_layers, batch_first=True)
        self.lstm = nn.GRU(project_dim * 3, lstm_hid_dim, lstm_num_layers, batch_first=True)
        self.lstm_projector = nn.Linear(lstm_hid_dim * lstm_num_layers, lstm_project_dim)

        matcher_dim = lstm_project_dim + project_dim
        self.matcher = nn.Linear(matcher_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, queries, crg):
        gz = crg.x
        gzs = []
        for conv, bn in zip(self.g_layers, self.g_batch_norms):
            gz = conv(gz, crg.edge_index, crg.edge_x)
            gz = F.relu(gz)
            gz = bn(gz)
            gzs.append(gz)
        gs = [global_add_pool(z, crg.batch) for z in gzs]
        g = torch.cat(gs, dim=1)
        g = self.g_projector(g)
        g = F.leaky_relu(g)
        
        seq = []
        for i in range(len(queries)):
            full_cat = []

            qz = queries[i].x
            qzs = []
            for conv, bn in zip(self.q_layers, self.q_batch_norms):
                qz = conv(qz, queries[i].edge_index, queries[i].edge_x)
                qz = F.relu(qz)
                qz = bn(qz)
                qzs.append(qz)
            qs = [global_add_pool(z, queries[i].batch) for z in qzs]
            q = torch.cat(qs, dim=1)
            q = self.q_projector(q)
            q = F.leaky_relu(q)
            full_cat.append(q)
            full_cat.append(q - g)
            full_cat.append(q * g)

            lstm_input = torch.cat(full_cat, dim=1)

            seq.append(lstm_input)
            # seq.append(q)

        seq = torch.stack(seq, dim=1)

        h0 = torch.zeros(self.lstm_num_layers, queries[0].batch_size, self.lstm_hid_dim).to(self.device)
        c0 = torch.zeros(self.lstm_num_layers, queries[0].batch_size, self.lstm_hid_dim).to(self.device)

        # _, (hn, cn) = self.lstm(seq, (h0, c0))
        _, hn = self.lstm(seq, h0)
        if self.lstm_num_layers > 1:
            to_cat = []
            for i in range(hn.shape[0]):
                to_cat.append(hn[i])
            hn_cat = torch.cat(to_cat, dim=1)
        else:
            hn_cat = hn[0]
            
        lstm_out = self.lstm_projector(hn_cat)
        lstm_out = F.leaky_relu(lstm_out)

        matcher_input = torch.cat([lstm_out, g], dim=1)

        output = self.matcher(matcher_input)
        output = self.sigmoid(output)

        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight) 
        m.bias.data.fill_(0.01)
        
class SeqDataset(Dataset):
    def __init__(self, seqs, lumps, y):
        super(SeqDataset, self).__init__()
        self.seqs = seqs
        self.lumps = lumps
        self.y = y

    def len(self):
        return len(self.y)
    
    def get(self, idx):
        seq = self.seqs[idx]
        lump = self.lumps[idx]
        return seq, lump, torch.tensor(self.y[idx], dtype=torch.float32)
    
class SeqTestDataset(Dataset):
    def __init__(self, seqs, lumps, ys):
        super(SeqTestDataset, self).__init__()
        self.seqs = seqs
        self.lumps = lumps
        self.ys = ys

    def len(self):
        return len(self.seqs)
    
    def get(self, idx):
        seq = self.seqs[idx]
        lump = self.lumps[idx]
        return seq, lump, torch.tensor(self.ys[idx], dtype=torch.long)


# %%
def strip_graph_attributes(graph):
    for (_, d) in graph.nodes(data=True):
        del d['pos']
    for (_, _, d) in graph.edges(data=True):
        del d['aid']

# %%
def class_seperation(y, graphs, min_size):
    classes = {}
    for c in set(y):
        classes[c] = []

    for i in range(len(y)):
        if graphs[i].number_of_nodes() >= min_size:
            classes[y[i]].append(i)
    
    the_keys = list(classes.keys())
    for c in the_keys:
        if len(classes[c]) == 0:
            del classes[c]

    return classes

def generate_lump_graphs(graphs, classes):
    lump_graphs = {}
    for c in classes:
        pos_feats = {}
        for i in classes[c]:
            graph = graphs[i]
            if graph is None:
                continue
            for node in graph.nodes():
                pos = graph.nodes[node]['pos']
                if not (pos[0] in pos_feats):
                    pos_feats[pos[0]] = {}
                if not (pos[1] in pos_feats[pos[0]]):
                    pos_feats[pos[0]][pos[1]] = []
                pos_feats[pos[0]][pos[1]].append(graph.nodes[node]['x'])

        pos_to_id_map = {}
        pos_id = 0
        for depth in pos_feats:
            pos_to_id_map[depth] = {}
            for order in pos_feats[depth]:
                pos_feats[depth][order] = np.array(pos_feats[depth][order], dtype=np.float32).mean(axis=0)
                pos_to_id_map[depth][order] = pos_id
                pos_id += 1

        pos_edge_feats = {}
        for i in classes[c]:
            graph = graphs[i]
            if graph is None:
                continue
            for edge in graph.edges():
                u = min(edge[0], edge[1])
                v = max(edge[0], edge[1])
                u_pos = graph.nodes[u]['pos']
                v_pos = graph.nodes[v]['pos']
                u_id = pos_to_id_map[u_pos[0]][u_pos[1]]
                v_id = pos_to_id_map[v_pos[0]][v_pos[1]]
                
                if not (u_id in pos_edge_feats):
                    pos_edge_feats[u_id] = {}
                if not (v_id in pos_edge_feats[u_id]):
                    pos_edge_feats[u_id][v_id] = []

                pos_edge_feats[u_id][v_id].append(graph.edges[edge[0], edge[1]]['x'])
        
        lump_graph_edges = []
        for u_id in pos_edge_feats:
            for v_id in pos_edge_feats[u_id]:
                pos_edge_feats[u_id][v_id] = np.array(pos_edge_feats[u_id][v_id], dtype=np.float32).mean(axis=0)
                lump_graph_edges.append((u_id, v_id, {"x":pos_edge_feats[u_id][v_id]}))

        node_attrs = {}
        for depth in pos_feats:
            for order in pos_feats[depth]:
                node_attrs[pos_to_id_map[depth][order]] = {"x":pos_feats[depth][order]}
                # node_attrs[pos_to_id_map[depth][order]] = {"x":pos_feats[depth][order], "pos":(depth, order)}
        
        lump_g = nx.from_edgelist(lump_graph_edges, create_using=nx.DiGraph)
        nx.set_node_attributes(lump_g, node_attrs)

        lump_graphs[c] = lump_g

    return lump_graphs

# %%
def generate_tree_sequences_train(graphs, seqs, aids, lump_graphs, y):
    aid_to_tree = {}
    aid_to_y = {}
    for i, aid in enumerate(aids):
        aid_to_tree[aid] = graphs[i]
        aid_to_y[aid] = y[i]

    tree_seqs = []
    lumps = []
    seq_y = []
    for seq in seqs:
        tree_seq = []
        for aid in seq:
            tree_seq.append(aid_to_tree[aid])
        
        c = np.argmax(aid_to_y[seq[-1]])
        for cc in lump_graphs:
            if cc != 'unseen':
                tree_seqs.append(tree_seq)
                lumps.append(from_networkx(lump_graphs[cc]))
                seq_y.append(1 if c == cc else 0)
    
    return tree_seqs, lumps, seq_y

# %%
def generate_tree_sequences_test(graphs, seqs, aids, lump_graphs, y):
    aid_to_tree = {}
    aid_to_y = {}
    for i, aid in enumerate(aids):
        aid_to_tree[aid] = graphs[i]
        aid_to_y[aid] = y[i]

    test_loaders = []
    seq_y = []
    for seq in seqs:
        tree_seq = []
        for aid in seq:
            tree_seq.append(aid_to_tree[aid])
        
        tree_seqs = []
        lumps = []
        ys = []
        c = aid_to_y[seq[-1]]
        for cc in lump_graphs:
            if cc != 'unseen':
                tree_seqs.append(tree_seq)
                lumps.append(from_networkx(lump_graphs[cc]))
                ys.append(cc)

        test_dataset = SeqTestDataset(tree_seqs, lumps, ys)
        test_loaders.append(DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False))
        seq_y.append(c)
    
    return test_loaders, seq_y

# %%
def make_directed(graphs):
    directed_graphs = []
    for i in range(len(graphs)):
        graph = graphs[i]
        strip_graph_attributes(graph)
        dg = graph.to_directed()
        to_remove = []
        for edge in dg.edges():
            if edge[0] > edge[1]:
                to_remove.append(edge)

        dg.remove_edges_from(to_remove)
        pyg = from_networkx(dg)
        if graph.number_of_nodes() == 1:
            pyg.edge_x = torch.empty((0, 20), dtype=torch.float)
        
        directed_graphs.append(pyg)

    return directed_graphs

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

def evaluate(model, test_dataloaders, test_y, k):
    model.eval()

    preds = []
    with torch.no_grad():
        for test_dataloader in test_dataloaders:
            for q, g, ys in test_dataloader:
                output = model(q, g).squeeze()
                indices = torch.topk(output, k, dim=0).indices.tolist()
                preds.append(ys[indices].tolist())

    corrects = [0] * len(test_y)
    pred_classes = [0] * len(test_y)
    mrrs = [0] * len(test_y)
    total = 0
    for i in range(len(test_y)):
        total += 1
        for j in range(len(preds[i])):
            if test_y[i] == preds[i][j]:
                corrects[i] = 1
                mrrs[i] = 1 / (j + 1)
    
    # f1 = f1_score(test_y, pred_classes, average=None)
    correct = sum(corrects)
    mrr = sum(mrrs)
    acc = round(correct / total, 4)
    mrr_acc = round(mrr / total, 4)
    return acc, mrr_acc #f1 #, correct, total, pred_classes

def calc_dst_probs(model, test_dataloaders):
    model.eval()
    probs = []
    with torch.no_grad():
        for test_dataloader in test_dataloaders:
            for q, g, ys in test_dataloader:
                output = model(q, g).squeeze()
                prob = output / output.sum()
                probs.append(prob.tolist())

    return probs


seed = str(sys.argv[1])
main_size = int(sys.argv[2])

tar_feats = None
select_k = 3
tar_feats = actcol_feats

test_id = [4]
trids = [[0], [0,1], [0,1,2], [0,1,2,3]]

with open(f'{data_path}/network_data/chunked_sessions/unbiased_seed_{seed}.pickle', 'rb') as fin:
    chunked_sessions = pickle.load(fin)

min_size = 1
train_x, train_seqs, train_y, train_aids, _, _, _, _ = treefy_sessions(
                                                                        sessions=chunked_sessions, 
                                                                        d_feats=display_pca_feats, 
                                                                        a_feats=concat_feats,
                                                                        tar=tar_feats, 
                                                                        min_size=min_size, 
                                                                        main_size=main_size,
                                                                        test_id=test_id,
                                                                        train_id=[0,1,2,3]
                                                                    )

non_hot_train_y = [int(np.argmax(train_y[i])) for i in range(len(train_y))]

train_classes = class_seperation(non_hot_train_y, train_x, main_size)
lump_graphs = generate_lump_graphs(train_x, train_classes)
unseen_lump = generate_lump_graphs(train_x, {'unseen':non_hot_train_y})
lump_graphs['unseen'] = unseen_lump['unseen']

train_x = make_directed(train_x)

log_size, avgs, stdevs = [], [], []
for trid in trids:
    
    _, my_train_seqs, _, _, test_x, test_seqs, test_y, test_aids = treefy_sessions(
                                                                                sessions=chunked_sessions, 
                                                                                d_feats=display_pca_feats, 
                                                                                a_feats=concat_feats,
                                                                                tar=tar_feats, 
                                                                                min_size=min_size, 
                                                                                main_size=main_size,
                                                                                test_id=test_id,
                                                                                train_id=trid
                                                                            )

    print(len(my_train_seqs), len(test_y))

    test_x = make_directed(test_x)

    test_loaders, test_seq_y = generate_tree_sequences_test(test_x, test_seqs, test_aids, lump_graphs, test_y)
    

    input_dim=181
    q_hid_dim=3000
    q_num_layers=1
    project_dim=3250
    output_dim=1
    lstm_hid_dim = 3500
    lstm_num_layers = 1
    lstm_project_dim = 3750 * 3 * lstm_num_layers

    ## seeding everything
    seed_everything(seed=get_truly_random_seed_through_os())
    model = MatchingNetwork(
                                input_dim=input_dim, 
                                q_hid_dim=q_hid_dim, 
                                q_num_layers=q_num_layers, 
                                project_dim=project_dim,
                                output_dim=output_dim,
                                lstm_hid_dim = lstm_hid_dim,
                                lstm_num_layers = lstm_num_layers,
                                lstm_project_dim = lstm_project_dim,
                                device=torch.device('cpu')
                            )
    # model.apply(init_weights)

    # Train model
    num_epochs = 5
    elapsed = []
    for epoch in range(1, num_epochs + 1):
        start = time.monotonic_ns()
        test_ra3, test_mrr = evaluate(model, test_loaders, test_seq_y, select_k)
        end = time.monotonic_ns()
        elapsed.append((end - start) / len(test_seq_y))

    avgs.append(statistics.mean(elapsed))
    stdevs.append(statistics.stdev(elapsed))
    log_size.append(len(my_train_seqs))
    
pickle.dump(
    {'log_sizes': log_size, 'avgs':avgs, 'stdevs':stdevs}, 
    open(f'{data_path}/network_data/model_stats/{seed}_{main_size}_gine_seq_logtime.pickle', 'wb'), 
    protocol=pickle.HIGHEST_PROTOCOL
)  
    




