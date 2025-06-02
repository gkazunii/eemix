#from pprint import pprint as pp
#from pprint import pformat as pf

import numpy as np
from collections import defaultdict

#NOTE N: number of not zero values
#NOTE D: number of dimensions

def reduce_1d_sum_sort(indices, values, d): #NOTE O(N)
    dims = [i for i in range( indices[0].shape[0] ) if i != d ]
    if len(values) <= 0: # if all zero tensor
        indices = np.array([])
        values = np.array([])
        return indices, values
    filter = make_filter(indices.shape[1], dims) #NOTE O(D)
    d = defaultdict(int)
    for k, v in zip(indices[:, filter], values): #NOTE O(N)
        d[tuple(k)] += v #NOTE O(1)
    indices = np.array(list(d.keys())).flatten()
    values = np.array(list(d.values()))
    
    sort_key = np.argsort(indices)
    indices = indices[sort_key]
    values = values[sort_key]
   
    return indices, values

def reduce_sum(indices, values, dims, sort=True): #NOTE O(N)
    if len(values) <= 0: # if all zero tensor
        indices = np.array([])
        values = np.array([])
        return indices, values
    filter = make_filter(indices.shape[1], dims) #NOTE O(D)
    d = defaultdict(int)
    for k, v in zip(indices[:, filter], values): #NOTE O(N)
        d[tuple(k)] += v #NOTE O(1)
    indices = np.array(list(d.keys()))
    values = np.array(list(d.values()))
   
    return indices, values


def reduce_sum_each_dim(indices, values, num_dims, sort=True): #NOTE O(ND)
    if len(values) <= 0: # if all zero tensor
        result_list = []
        for d in range(num_dims):
            indices = np.array([])
            values = np.array([])
            result_list.append((indices, values))
        return result_list
    box = []
    for _ in range(num_dims): #NOTE O(D)
        box.append(defaultdict(int)) #NOTE O(1)
    for one_index, v in zip(indices, values): #NOTE O(N)
        for d, k in enumerate(one_index): #NOTE O(D)
            box[d][k] += v #NOTE O(1)
    result_list = []
    for d in box: #NOTE O(D)
        indices = np.array(list(d.keys()))
        values = np.array(list(d.values()))
        
        if sort:
            sort_key = np.argsort(indices)
            indices = indices[sort_key]
            values = values[sort_key] 
        
        result_list.append((indices, values))
    return result_list

def make_filter(index_length, dims):
    filter = [True] * index_length
    for d in dims:
        filter[d] = False
    return filter

