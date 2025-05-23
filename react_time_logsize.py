import numpy as np
import time
import sys
import pickle
import math
import traceback
from functools import partial
from collections import Counter
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from lib.utilities import Repository
from sklearn.neighbors import BallTree
import zss



repo = Repository('./session_repositories/actions.tsv','./session_repositories/displays.tsv','./raw_datasets/')


with open(f'./edge/act_five_feats.pickle', 'rb') as fin:
    act_feats = pickle.load(fin)

with open(f'./edge/col_action.pickle', 'rb') as fin:
    col_feats = pickle.load(fin)

with open(f'./display_feats/display_feats.pickle', 'rb') as fin:
    display_feats = pickle.load(fin)

with open(f'./display_feats/display_pca_feats.pickle', 'rb') as fin:
    display_pca_feats = pickle.load(fin)

together_feats = {}
for key in act_feats:
    feat = np.zeros(len(act_feats[key]) * len(col_feats[key]))
    offset = np.argmax(act_feats[key]) * len(col_feats[key])
    feat[offset + np.argmax(col_feats[key])] = 1
    together_feats[key] = feat.copy()

concat_feats = {}
for key in act_feats:
    concat_feats[key] = np.concatenate((act_feats[key], col_feats[key])).copy()

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


class WeirdNode(object):
    def __init__(self, my_type, my_id, my_x):
        self.my_type = my_type
        self.my_id = my_id
        self.my_x = my_x
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node.my_children

    def get_type(self):
        return self.my_type

    def get_id(self):
        return self.my_id

    def get_x(self):
        return self.my_x

    def addkid(self, node, before=False):
        if before:  self.my_children.insert(0, node)
        else:   self.my_children.append(node)
        return self


def replay_graph(edges, d_feats, a_feats, tar, min_size, max_size):
    logic_error_displays = [427, 428, 429, 430, 
                        854, 855, 856, 868, 891, 
                        977, 978, 979, 980, 
                        1304, 1908, 1909, 1983, 
                        2022, 2023, 2024, 2195,
                        3244, 3446, 3447, 
                        4050, 4051, 4056, 4052, 4054, 4055, 4057, 4058, 4059, 
                        4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067]

    action_index = {'project':1, 'filter':2, 'group':3, 'sort':4}
    
    replays = []
    y = []
    end_aids = []
    try:
        BigG = nx.from_edgelist(edges, create_using=nx.Graph)
        S = [BigG.subgraph(c).copy() for c in nx.connected_components(BigG)]

        for G in S:
            g_nodes = list(G.nodes())
            g_nodes.sort()
            g_nodes.reverse()
            for end in G.nodes():
                if end in logic_error_displays:
                    continue
                
                for context in range(min_size, max_size + 1):
                    tree_nodes = set()
                    for v in g_nodes:
                        if v < end and len(tree_nodes) < context:
                            tree_nodes.add(v)

                    if len(tree_nodes) < context:
                        continue

                    leading_to_end = get_predecessors(G, end)[0]
                    # tree_nodes.add(leading_to_end)
                    
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

                    g_context = nx.induced_subgraph(G, tree_nodes).copy()
                    # print(g_context.number_of_nodes())

                    root = None
                    for v in g_context.nodes():
                        if len(get_predecessors(g_context, v)) == 0:
                            root = v

                    node_order, node_count_at_depth, nodes_at_depth = bfs(g_context, root)

                    node_depth = {}
                    for depth in nodes_at_depth:
                        for v in nodes_at_depth[depth]:
                            node_depth[v] = depth

                    # print(node_depth)

                    edges_to_remove = []
                    g_context = g_context.to_directed()
                    for edge in g_context.edges():
                        if edge[0] > edge[1]:
                            edges_to_remove.append((edge[0], edge[1]))
                    g_context.remove_edges_from(edges_to_remove)

                    zss_context = {}
                    for edge in g_context.edges():
                        u = edge[0]
                        v = edge[1]
                        aid = g_context.edges[u, v]['aid']
                        naid = 10000 + aid

                        if u not in zss_context:
                            zss_context[u] = WeirdNode('display', u, d_feats[u])
                        if v not in zss_context:
                            zss_context[v] = WeirdNode('display', v, d_feats[v])
                        if naid not in zss_context:
                            zss_context[naid] = WeirdNode('action', aid, a_feats[aid])
                        
                        zss_context[u].addkid(zss_context[naid])
                        zss_context[naid].addkid(zss_context[v])

                    end_aid = G.edges[leading_to_end, end]['aid']
                    replays.append((zss_context, root))
                    target = []
                    for t in tar:
                        target.append(np.argmax(t[end_aid]))
                    y.append(target)
    except Exception as e:
        print(g_context.edges.data())
        print(traceback.format_exc())

    return replays, y


def treefy_sessions(sessions, d_feats, a_feats, tar, min_size, max_size, test_id, train_id):
    test_contexts = []
    test_y = []
    train_contexts = []
    train_y = []
    for chunk_id in sessions:
        if chunk_id in test_id:
            for edges in sessions[chunk_id]:
                g_contexts, gys = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, max_size=max_size)
                test_contexts.extend(g_contexts)
                test_y.extend(gys)
        elif chunk_id in train_id:
            for edges in sessions[chunk_id]:
                g_contexts, gys = replay_graph(edges, d_feats, a_feats, tar, min_size=min_size, max_size=max_size)
                train_contexts.extend(g_contexts)
                train_y.extend(gys)

    return train_contexts, train_y, test_contexts, test_y


def generate_graphs(repo, d_feats, a_feats, tar, size, test_project):
    og_columns = ['captured_length', 'length', 'tcp_stream', 'number', 'eth_dst', 'eth_src', 
                'highest_layer', 'info_line', 'interface_captured', 'ip_dst', 'ip_src', 'sniff_timestamp', 'tcp_dstport', 'tcp_srcport']
    init_display_ids = {1:1, 2:1, 3:1, 4:1}
    train_seq = []
    train_y = []
    train_contexts = []
    train_fullG = []
    train_session = []
    test_seq = []
    test_y = []
    test_contexts = []
    test_fullG = []
    test_session = []
    for project_id in range(1, 5):
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

            if unrelated:
                continue

            if project_id == test_project:
                g_replays, g_ys, g_contexts, full_gs = replay_graph(edges, d_feats, a_feats, tar, size=size, is_test=True)
                test_seq.extend(g_replays)
                test_y.extend(g_ys)
                test_contexts.extend(g_contexts)
                test_fullG.extend(full_gs)
                test_session.extend([session_id] * len(g_ys))
            else:
                g_replays, g_ys, g_contexts, full_gs = replay_graph(edges, d_feats, a_feats, tar, size=size, is_test=False)
                train_seq.extend(g_replays)
                train_y.extend(g_ys)
                train_contexts.extend(g_contexts)
                train_fullG.extend(full_gs)
                train_session.extend([session_id] * len(g_ys))
    return train_seq, train_y, train_contexts, train_fullG, train_session, test_seq, test_y, test_contexts, test_fullG, test_session


def node_ins_cost(node):
    return 1.0

def node_del_cost(node):
    return 1.0

def node_update_cost(node1, node2):
    if node1.get_type() == node2.get_type():
        if node1.get_type() == 'display':
            return repo.display_distance(node1.get_id(), node2.get_id())
        elif node1.get_type() == 'action':
            return repo.action_distance(node1.get_id(), node2.get_id())
    else:
        return 2.0 #non-simillar nodes account for deletion and insertion

def calculate_ted(g1_index, g2_index, g_list):
    g1_index = math.ceil(g1_index[0])
    g2_index = math.ceil(g2_index[0])

    # print(g1_index, g2_index)

    g1_tuple = g_list[g1_index]
    g2_tuple = g_list[g2_index]
    g1, g1_root = g1_tuple[0], g1_tuple[1]
    g2, g2_root = g2_tuple[0], g2_tuple[1]
    ted = zss.distance(g1[g1_root], g2[g2_root], WeirdNode.get_children, node_ins_cost, node_del_cost, node_update_cost)
    return ted


def generate_display_features(tid):
    feats_for_pca = []
    for d in display_feats:
        pid = repo.displays[repo.displays['display_id'] == d]['project_id'].item()
        if pid != tid:
            feats_for_pca.append(display_feats[d])
        
    feats_for_pca = np.array(feats_for_pca)
    big_pca = PCA(n_components=min(feats_for_pca.shape[0], feats_for_pca.shape[1]))
    big_pca.fit(feats_for_pca)

    total_evr = 0.0
    components = 0
    for evr in big_pca.explained_variance_ratio_:
        total_evr += evr
        components += 1
        if total_evr >= 0.99:
            break

    print(total_evr, components)

    pca = PCA(n_components=components)
    pca.fit(feats_for_pca)
    pca_scaler = MinMaxScaler()
    pca_scaler.fit(pca.transform(feats_for_pca))

    display_pca_feats = {}
    for d in display_feats:
        display_pca_feats[d] = pca_scaler.transform(pca.transform([display_feats[d]]))[0]

    return display_pca_feats


def calculate_test_ra3(preds, test_y):
    accs = []
    for task in range(3):
        corrects = [0] * len(test_y[:, task])
        total = 0
        for i in range(len(test_y[:, task])):
            total += 1
            for j in range(len(preds[task][i])):
                pred = preds[task][i][j][0]
                if pred == test_y[:, task][i]:
                    corrects[i] = 1

        correct = sum(corrects)
        acc = round(correct / total, 4)
        accs.append(acc)

    return accs


def calculate_test_mrr(preds, test_y):
    mrr_accs = []
    for task in range(3):
        mrrs = [0] * len(test_y[:, task])
        total = 0
        for i in range(len(test_y[:, task])):
            total += 1
            for j in range(len(preds[task][i])):
                pred = preds[task][i][j][0]
                if pred == test_y[:, task][i]:
                    mrrs[i] = (1 / (j + 1))
        
        mrr = sum(mrrs)
        mrr_acc = round(mrr / total, 4)
        mrr_accs.append(mrr_acc)

    return mrr_accs

seed = int(sys.argv[1])
size = int(sys.argv[2])
# raw_tid = int(sys.argv[3])
tid = [4]
trids = [[0], [0,1], [0,1,2], [0,1,2,3]]

chunked_sessions = pickle.load(open(f'./chunked_sessions/unbiased_seed_{seed}.pickle', 'rb'))

# if os.path.isfile(f'./chunk_ted_results/{seed}_{size}_{tid}_unbiased.pickle'):
#     sys.exit("Already exists.")
log_size, elapsed = [], []
for trid in trids:
    train_g_og, train_y_og, test_g_og, test_y_og = treefy_sessions(
                                                                        sessions=chunked_sessions, 
                                                                        d_feats=display_pca_feats, 
                                                                        a_feats=concat_feats,
                                                                        tar=[act_feats, col_feats, together_feats], 
                                                                        min_size=3, 
                                                                        max_size=size,
                                                                        test_id=tid,
                                                                        train_id=trid
                                                                    )  


    k_mid = int(math.ceil(math.sqrt(len(train_g_og))))
    k_start = int(math.floor(k_mid - (k_mid / 2)))
    k_limit = int(math.ceil(k_mid + (k_mid / 2)))

    g_list = []
    g_list.extend(train_g_og)
    g_list.extend(test_g_og)

    train_y = np.array(train_y_og)
    test_y = np.array(test_y_og)

    test_offset = len(train_g_og)

    partial_calculate_ted = partial(calculate_ted, g_list=g_list)
    x = np.array(list(range(len(train_g_og))), dtype=np.int32).reshape(-1, 1)
    balltree = BallTree(x, leaf_size=30, metric=partial_calculate_ted)

    test_close = [[], [], []]
    # with tqdm(total=len(test_g_og), desc='(T)') as pbar:
    start_time = time.monotonic_ns()
    for i in range(len(test_g_og)):
        _, indices = balltree.query([[test_offset + i]], k=k_mid)
        for task in range(3):
            test_close[task].append(train_y[:, task][indices[0,:]])
        # pbar.update()
    end_time = time.monotonic_ns()
    elapsed.append((end_time - start_time) / len(test_g_og))
    log_size.append(len(train_g_og))


    ra3_preds = [[], [], []]
    mrr_preds = [[], [], []]
    for task in range(3):
        for matching_classes in test_close[task]:
            if task == 0:
                n = 1
            else:
                n = 3
            most_common = Counter(matching_classes[:k_mid]).most_common()
            freq_most_common = [item for item in most_common if (item[1] / k_mid) >= 0.1]
            top_classes = freq_most_common[:min(n, len(freq_most_common))]
            ra3_preds[task].append(top_classes)

            most_common = Counter(matching_classes[:k_mid]).most_common()
            freq_most_common = [item for item in most_common if (item[1] / k_mid) >= 0.1]
            top_classes = freq_most_common[:min(n, len(freq_most_common))]
            mrr_preds[task].append(top_classes)

    ra3_accs = calculate_test_ra3(ra3_preds, test_y)
    mrr_accs = calculate_test_mrr(mrr_preds, test_y)

    print('seed =', seed, 'size =', size, 'tid =', tid)

pickle.dump(
    {'log_sizes':log_size, 'time':elapsed}, 
    open(f'./chunk_ted_results/{seed}_{size}_time.pickle', 'wb'), 
    protocol=pickle.HIGHEST_PROTOCOL
)

        


