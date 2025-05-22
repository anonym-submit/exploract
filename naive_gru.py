# %%
import sys
import numpy as np
import random
import pickle
import traceback
import networkx as nx
import torch
import torch.nn.functional as F
from lib.utilities import Repository
import os
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from livelossplot import PlotLosses

data_path = '/data/gpfs/projects/punim2258'

# %%
repo = Repository('./session_repositories/actions.tsv','./session_repositories/displays.tsv','./raw_datasets/')

# %%
device = torch.device('cuda')

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
                    seq.append(torch.tensor(tar[v_aid], dtype=torch.float32))

                leading_to_end = get_predecessors(G, end)[0]
                end_aid = G.edges[leading_to_end, end]['aid']
                
                end_aids.append(end_aid)

                if reported_size == main_size:
                    seq.append(torch.tensor(tar[end_aid], dtype=torch.float32))
                
                if len(seq) > 0 and reported_size == main_size:
                    seqs.append(seq)

                    if is_train:
                        target = tar[end_aid]
                    else:
                        target = np.argmax(tar[end_aid])
                    y.append(target)

                
    except Exception as e:
        # print(g_context.edges.data())
        print(traceback.format_exc())

    return seqs, end_aids, y

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
def treefy_sessions(sessions, d_feats, a_feats, tar, min_size, main_size, test_id):
    test_seqs = []
    test_aids = []
    test_y = []
    train_seqs = []
    train_aids = []
    train_y = []
    for chunk_id in sessions:
        if chunk_id in test_id:
            for edges in sessions[chunk_id]:
                seqs, aids, g_ys = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, main_size=main_size, is_train=False)
                test_seqs.extend(seqs)
                test_aids.extend(aids)
                test_y.extend(g_ys)
        else:
            for edges in sessions[chunk_id]:
                seqs, aids, g_ys = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, main_size=main_size, is_train=True)
                train_seqs.extend(seqs)
                train_aids.extend(aids)
                train_y.extend(g_ys)

    return train_seqs, train_aids, train_y, test_seqs, test_aids, test_y

class MatchingNetwork(nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, lstm_hid_dim, lstm_num_layers, lstm_project_dim, device):
        super(MatchingNetwork, self).__init__()

        self.device = device

        self.projector = nn.Linear(input_dim, project_dim)

        self.lstm_hid_dim = lstm_hid_dim
        self.lstm_num_layers = lstm_num_layers
        # self.lstm = nn.LSTM(project_dim * 1, lstm_hid_dim, lstm_num_layers, batch_first=True)
        self.lstm = nn.GRU(project_dim * 1, lstm_hid_dim, lstm_num_layers, batch_first=True)
        self.lstm_projector = nn.Linear(lstm_hid_dim * lstm_num_layers, lstm_project_dim)

        matcher_dim = lstm_project_dim #* self.lstm_num_layers
        self.matcher = nn.Linear(matcher_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, queries):
        seq = []
        for i in range(len(queries)):
            full_cat = []

            q = self.projector(queries[i])
            q = F.leaky_relu(q)
            full_cat.append(q)
            lstm_input = torch.cat(full_cat, dim=1)

            seq.append(lstm_input)
            # seq.append(q)

        seq = torch.stack(seq, dim=1)

        h0 = torch.zeros(self.lstm_num_layers, queries[0].shape[0], self.lstm_hid_dim).to(self.device)
        # c0 = torch.zeros(self.lstm_num_layers, queries[0].batch_size, self.lstm_hid_dim).to(self.device)

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

        output = self.matcher(lstm_out)
        output = self.sigmoid(output)

        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.weight) 
        m.bias.data.fill_(0.01)
        
class SeqDataset(Dataset):
    def __init__(self, seqs, y):
        super(SeqDataset, self).__init__()
        self.seqs = seqs
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq, torch.tensor(self.y[idx], dtype=torch.float32)
    
class SeqTestDataset(Dataset):
    def __init__(self, seqs, ys):
        super(SeqTestDataset, self).__init__()
        self.seqs = seqs
        self.ys = ys

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq, torch.tensor(self.ys[idx], dtype=torch.long)
    

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

with open(f'{data_path}/network_data/chunked_sessions/unbiased_seed_{seed}.pickle', 'rb') as fin:
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

min_size = 1
train_seqs, train_aids, train_y, test_seqs, test_aids, test_y = treefy_sessions(
                                                                    sessions=chunked_sessions, 
                                                                    d_feats=display_pca_feats, 
                                                                    a_feats=concat_feats,
                                                                    tar=tar_feats, 
                                                                    min_size=min_size, 
                                                                    main_size=main_size,
                                                                    test_id=test_id
                                                                )
print(len(train_seqs), len(train_y))
print(len(test_seqs), len(test_y))

train_seqs_by_length = {}
train_seq_y_by_length = {}
for i, seq in enumerate(train_seqs):
    if len(seq) in train_seqs_by_length:
        train_seqs_by_length[len(seq)].append(seq)
        train_seq_y_by_length[len(seq)].append(train_y[i])
    else:
        train_seqs_by_length[len(seq)] = [seq]
        train_seq_y_by_length[len(seq)] = [train_y[i]]

train_dataloaders = []
for length in train_seqs_by_length:
    train_dataset = SeqDataset(train_seqs_by_length[length], train_seq_y_by_length[length])
    train_dataloaders.append(DataLoader(train_dataset, batch_size=max(2, len(train_y)//8), shuffle=True))

# train_dataloaders.reverse()

test_seqs_by_length = {}
test_seq_y_by_length = {}
for i, seq in enumerate(test_seqs):
    if len(seq) in test_seqs_by_length:
        test_seqs_by_length[len(seq)].append(seq)
        test_seq_y_by_length[len(seq)].append(test_y[i])
    else:
        test_seqs_by_length[len(seq)] = [seq]
        test_seq_y_by_length[len(seq)] = [test_y[i]]

test_dataloaders = []
for length in test_seqs_by_length:
    test_dataset = SeqTestDataset(test_seqs_by_length[length], test_seq_y_by_length[length])
    test_dataloaders.append(DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False))

# %%
def evaluate(model, test_dataloaders, test_y, k):
    model.eval()

    preds = []
    with torch.no_grad():
        for test_dataloader in test_dataloaders:
            for q, ys in test_dataloader:
                q = [qi.to(device) for qi in q]
                output = model(q)
                indices = torch.topk(output, k, dim=1).indices.tolist()
                preds.extend(indices)

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
    
    correct = sum(corrects)
    mrr = sum(mrrs)
    acc = round(correct / total, 4)
    mrr_acc = round(mrr / total, 4)
    return acc, mrr_acc #f1 #, correct, total, pred_classes

# %%
input_dim=tar_feats[1].shape[0]
project_dim=3250
output_dim=tar_feats[1].shape[0]
lstm_hid_dim = 3500
lstm_num_layers = 1
lstm_project_dim = 3750 * 1 * lstm_num_layers


results = {'ra3':[], 'mrr':[]}
for _ in range(5):
    ## seeding everything
    seed_everything(seed=get_truly_random_seed_through_os())
    model = MatchingNetwork(
                                input_dim=input_dim, 
                                project_dim=project_dim,
                                output_dim=output_dim,
                                lstm_hid_dim = lstm_hid_dim,
                                lstm_num_layers = lstm_num_layers,
                                lstm_project_dim = lstm_project_dim,
                                device=device
                            ).to(device)
    # model.apply(init_weights)

    criterion = nn.BCELoss()

    # model.load_state_dict(torch.load(f'./gine_seq_crg_saved/gine_seq_crg_{task}_best_ra3.pt', weights_only=True))

    # %%
    learning_rate = 1e-4
    wd = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    peak_ra3 = 0.0
    peak_mrr = 0.0

    # Train model
    num_epochs = 200

    max_ra3 = 0.0
    max_mrr = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for dataloader in train_dataloaders:
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
        
        test_ra3, test_mrr = evaluate(model, test_dataloaders, test_y, select_k)

        if test_ra3 > max_ra3:
            max_ra3 = test_ra3
        
        if test_mrr > max_mrr:
            max_mrr = test_mrr

    results['ra3'].append(max_ra3)
    results['mrr'].append(max_mrr)
        
pickle.dump(
    results, 
    open(f'{data_path}/network_data/model_stats/{task}_{seed}_{main_size}_{test_id}_hot_lstm.pickle', 'wb'), 
    protocol=pickle.HIGHEST_PROTOCOL
)  

