from pymatgen.analysis.local_env import CrystalNN,EconNN
import numpy as np
import torch

def get_neighbors_crystalnn(structure):
    cnn = CrystalNN()
    enn = EconNN()
    site_all_list, site_num_list = [], []
    row, col, distances,edge_vec = [], [], [], []
    
    #nnum = sum(structure.natoms)
    for i in range(len(structure.sites)):
        start =i
        center_site = np.array(structure[i].coords)
        crystalnn = cnn.get_nn_info(structure,i)
        site_num = len(crystalnn)
        if site_num == 0 :
            print("Choose enn methods..")
            crystalnn = enn.get_nn_info(structure,i)
        
        for atom in crystalnn:
            end = atom['site_index']
            end_coords = np.array(atom['site'].coords,dtype=object)
            row += [start]
            col += [end]
            edge_vec_t = np.array(center_site) - np.array(end_coords)
            edge_vec.append(edge_vec_t)
            distances.append(np.array(atom['site'],dtype=object)[1])
            #print(distance)
            #distances += [distance]

        #exit()

        #site_num_list.append(site_num)
        #one_site_list = one_site_process(crystalnn)
        #site_all_list.append(one_site_list)
        #edge_index = torch.tensor([row,col], dtype=torch.long)
        #perm = (edge_index[0] * site_num + edge_index[1]).argsort()
        edge_index = [row,col]
    return  edge_index, distances, edge_vec

def one_site_process(crystalnn):
    site_list, site_list_t = [],[]
    for i in range(len(crystalnn)):
        for k in crystalnn[i].items():
            if k[0] == 'site':
                site_list_t = list(k[1:])
                site_list.append(site_list_t[0])
    return site_list


