from pymatgen.analysis.local_env import CrystalNN,MinimumDistanceNN
import numpy as np
import torch

#defining periodic graph within cutoff.
def get_radius_graph(structure, cutoff, max_neighbors):
   # cnn = CrystalNN()
    MNN = MinimumDistanceNN(cutoff=cutoff,get_all_sites=True)
    edge_src, edge_dest, edge_vec, distance = [], [], [], []
    #enn = EconNN()
    #site_all_list, site_num_list = [], []
    #row, col, distances = [], [], []
    
    #nnum = sum(structure.natoms)
    for i in range(len(structure.sites)):
        start =i
        center_site = np.array(structure[i].coords)
        mdnn = MNN.get_nn_info(structure,i)
        #site_num = len(mdnn)
        
        for atom in mdnn:
            end = atom['site_index']
            end_coords = np.array(atom['site'].coords,dtype=object)
            edge_src += [start]
            edge_dest += [end]
            edge_vec_t = np.array(center_site) - np.array(end_coords)
            edge_vec.append(edge_vec_t)
           # print(np.array(atom['site'],dtype=object)[1])
            distance.append(np.array(atom['site'],dtype=object)[1])
    
    edge_src, edge_dest, edge_vec, distance = np.array(edge_src), np.array(edge_dest), np.array(edge_vec), np.array(distance)   
    
    max_neigh_index = []
    for i in range(len(structure.sites)):
        idx_i = (edge_src == i).nonzero()[0]
        idx_sorted = np.argsort(distance[idx_i])[: max_neighbors]
        max_neigh_index.append(idx_i[idx_sorted])

    max_neigh_index=np.concatenate(max_neigh_index)

    edge_src, edge_dest, edge_vec, distance=edge_src[max_neigh_index], edge_dest[max_neigh_index], edge_vec[max_neigh_index], distance[max_neigh_index]
    return edge_src, edge_dest, edge_vec, distance

