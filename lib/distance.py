import pandas as pd

##########################
#   DISPLAY SIMILARITY   #
##########################


def column_data_difference(col_dict1, col_dict2):
    size = len(col_dict1)
    diff = 0
    for (_,v1), (_,v2) in zip(col_dict1.items(), col_dict2.items()):
        if v1 == v2:
            continue
        else:
            diff += 1 / size - min(v1, v2) / (size * max(v1, v2))
    return diff
        
        
def data_distance(dl1, dl2, scale=True):

    # Set Difference:
    at1 = set(dl1.keys())
    at2 = set(dl2.keys())
    ikeys = at1.intersection(at2)
    diff_size = len(at1.union(at2) - ikeys)

    raw_dist = diff_size
    for k in ikeys:
        raw_dist += column_data_difference(dl1[k], dl2[k])

    return raw_dist if not scale else (2 * raw_dist) / (len(at1) + len(at2) + raw_dist)


def gran_distance(gl1, gl2, scale=True):

    group_attr1 = set(gl1['group_attrs'])
    group_attr2 = set(gl2['group_attrs'])
    group_diff_size = len(group_attr1.union(group_attr2) - group_attr1.intersection(group_attr2))
    agg_attr1 = set(gl1['agg_attrs'].keys())
    agg_attr2 = set(gl2['agg_attrs'].keys())
    agg_intersection = agg_attr1.intersection(agg_attr2)
    agg_diff_size = len(agg_attr1.union(agg_attr2) - agg_intersection)
    agg_nve_diff = agg_diff_size
    for agg_column in agg_intersection:
        v1 = gl1['agg_attrs'][agg_column]
        v2 = gl2['agg_attrs'][agg_column]
        if v1 == v2:
            continue
        else:
            agg_nve_diff = 1 - min(v1, v2) / max(v1, v2)

    col_gran_distance = group_diff_size + agg_nve_diff
    normed_gran_distance = 2 * col_gran_distance / (len(group_attr1)
                                                    + len(group_attr2)
                                                    +len(agg_attr1)
                                                    +len(agg_attr2)
                                                    +col_gran_distance)

    meta_score = 0
    measures = ['ngroups', 'size_mean', 'size_var']
    for m in measures:
        if gl1[m] == gl2[m]:
            continue
        else:
            meta_score += 1 / len(measures) -  min(gl1[m],gl2[m]) / (len(measures) * max(gl1[m],gl2[m]))

    return ((normed_gran_distance+meta_score)/2)
    #print(agg_diff_size,group_diff_size)
    

def display_distance(disp_data1, disp_data2, method="all"):
    # print(disp1.display_id,disp2.display_id)
    data1 = disp_data1["data_layer"]
    data2 = disp_data2["data_layer"]
    data_dist = data_distance(data1, data2)
    if "ds_only" in method:
        return data_dist

    gran1 = disp_data1["granularity_layer"]
    gran2 = disp_data2["granularity_layer"]

    if pd.isnull(gran1) and pd.isnull(gran2):
        granularity_distance = 0.0
    elif pd.isnull(gran1) or pd.isnull(gran2):
        granularity_distance = 1.0
    elif (not bool(gran1)) and (not bool(gran2)):
        granularity_distance = 0.0
    elif (not bool(gran1)) or (not bool(gran2)):
        granularity_distance = 1.0
    else:
        granularity_distance = gran_distance(gran1, gran2)

    return (data_dist + granularity_distance) / 2


#########################
#   ACTION SIMILARITY   #
#########################


def pair_lca(pair1, pair2):

    k1, v1 = pair1
    k2, v2 = pair2
    if k1 == k2:
        if v1 == v2:
            return (k1, v1)
        else:
            return (k1, None)
    else:
        if v1 == v2:
            return (None, v2)
        else:
            return (None, None)


def set_lca(set1, set2):
    lca = set()
    for pair1 in set1:
        for pair2 in set2:
            p_lca = pair_lca(pair1, pair2)
            if p_lca != (None, None):
                lca.add(p_lca)

    lca_temp = set(lca)

    for pair1 in lca_temp:
        for pair2 in lca_temp:
            if pair1 == pair2:
                continue

            if is_pair_more_general_or_equal(pair1, pair2):
                lca.discard(pair1)

    return lca


def is_pair_more_general_or_equal(pair1, pair2):
    k1, v1 = pair1
    k2, v2 = pair2
    if k1 == k2 and v1 == v2:
        return True
    if k1 is None and v1 is None:
        return True
    if k1 == k2 and v1 is None:
        return True
    if v1 == v2 and k1 is None:
        return True
    return False


def set_dist(set1, set2):
    if len(set2) is not 0:
        K2 = set([x for (x, _) in set2])
        V2 = set([x for (_, x) in set2])
    else:
        K2 = []
        V2 = []
    dist_sum = 0
    for pair1 in set1:
        if pair1 in set2:
            continue
        k1, v1 = pair1
        if k1 in K2:
            if v1 in V2:
                dist_sum += 1
            else:
                dist_sum += 2
        elif v1 in V2:
            dist_sum += 2
        else:
            dist_sum += 3

    return dist_sum


def action_distance(set1, set2, verbose=False):
    s_lca = set_lca(set1, set2)
    if verbose:
        print("LCA:", s_lca)
    d1 = set_dist(set1, s_lca)
    d2 = set_dist(set2, s_lca)
    d3 = set_dist(set1, [])
    d4 = set_dist(set2, [])

    dist = (d1 + d2) / (d3 + d4)
    if verbose:
        print("1 to lca:", d1, "\n2 to lca:", d2, "\n1 to root:", d3, "\n2 to root:", d4)
    return dist

