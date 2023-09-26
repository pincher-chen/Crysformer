from typing import List
from features.atom_feat import AtomCustomJSONInitializer,GaussianDistance
from pymatgen.core.structure import Structure
import numpy as np
import json
import pymatgen as mg
import os
import pickle
import torch
from ase.io import read
import ase
import ase.neighborlist
from io import StringIO

#from features.get_neighbors_crystalnn import get_neighbors_crystalnn
#from features.get_radius_graph_cutoff import get_radius_graph
#from features.get_radius_graph_cutoff_knn import get_radius_graph_knn

from torch_geometric.data import Data, InMemoryDataset, download_url
import os.path as osp
from tqdm import tqdm 

class MP(InMemoryDataset):
    
    raw_url = 'http://figshare.com/ndownloader/files/15087992'

    def __init__(self, root, split, feature_type="crystalnet", fixed_size_split=True):
        assert feature_type in ["crystalnet"], "Please use valid features"
        assert split in ["train", "valid", "test"]
        self.split = split
        self.root = osp.abspath(root)
        self.feature_type = feature_type
        self.fixed_size_split = fixed_size_split

        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean))) #median absolute deviation
        return mean, mad
    

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())


    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import pymatgen  # noqa
            return ['mp.2018.6.1.json']
        except ImportError:
            return ['mp_crystalnet.pt']
    @property
    def processed_file_names(self) -> str:
        return "_".join([self.split, self.feature_type]) + '.pt'

    def download(self):
        try:
            file_path = download_url(self.raw_url, self.raw_dir)

            os.rename(osp.join(self.raw_dir, '15087992'),osp.join(self.raw_dir, 'mp.2018.6.1.json'))
        except ImportError:
            print("No raw files find.")
 
    def process(self):
        data_path ='./conf'
        ari = AtomCustomJSONInitializer(f'{data_path}/atom_init.json')
        dmin = 0
        dmax = 5
        step = 0.1
        var = 0.5
        radius = 5
        max_num_neighbors = 16

        gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)

        #data = json.load(open('/GPUFS/nscc-gz_pinchen2/apps/deepLearning/pytorch/CrystalNet/data/MP/mp.2018.6.1.json'))
        #data = json.load(open('self.raw_url/mp.2018.6.1.json'))
        data_source = json.load(open(self.raw_paths[0]))
        N_mat = len(data_source)

        #train-validation-test split of 60,000-5000-4239

        N_test = 4239
        N_val =  5000
        N_train = N_mat - (N_test + N_val)
        N_mat_t = 6000#6000

        if self.fixed_size_split:
            N_test = 4239
            N_val = 5000
            N_train = N_mat - (N_test + N_val)
            data_perm = np.random.default_rng(1).permutation(N_mat)

        train, valid, test = np.split(data_perm, [N_train, N_train+N_val])
        #print(train)
        #print(valid)
        #print(test)
        indices = {"train": train, "valid": valid, "test": test}
        failed_list = ['mp-994911']

        np.savez(os.path.join(self.root, 'splits.npz'), idx_train=train, idx_valid=valid, idx_test=test)

        data_list = []

        j = 0
        #target = []
        #for i in range(0,N_mat):
        for i, mat in enumerate(tqdm(data_source)):
            if j not in indices[self.split]:
                j += 1
                continue
            j += 1  
          
            file_name = mat['material_id']
            if file_name in failed_list:
                #print(file_name + "fail to")
                continue
            #print(file_name)
            target = [mat['formation_energy_per_atom'],mat['band_gap']]
            #print(target_t)
            #target += target_t
            #target = mat['formation_energy_per_atom']
            y = torch.tensor(target)
            #print(y)
            y = y.unsqueeze(0)
            #print(y)
            crystal = Structure.from_str(mat['structure'],fmt='cif')
            #from ase.io import read
            file_io = StringIO(mat['structure'])
            crystal2 = ase.io.read(file_io,format='cif')
            #print(crystal2)
            num_nodes = len(crystal)
            node_index = torch.tensor([i for i in range(num_nodes)])

            #pos = [crystal[atom].coords for atom in range(len(crystal)) ]
            #pos = torch.tensor(pos, dtype=torch.float)

            #build pair-wise edge graphs
            #edge_d_dst_index = torch.repeat_interleave(node_index, repeats=num_nodes)
            #edge_d_src_index = node_index.repeat(num_nodes)
            #edge_d_attr = pos[edge_d_dst_index] - pos[edge_d_src_index]
            #edge_d_attr = edge_d_attr.norm(dim=1, p=2)
            #edge_d_dst_index = edge_d_dst_index.view(1, -1)
            #edge_d_src_index = edge_d_src_index.view(1, -1)
            #edge_d_index = torch.cat((edge_d_dst_index, edge_d_src_index), dim=0)
   
            type_idx = []
            atomic_num = [crystal[j].specie.number for j in range(len(crystal))]
            z = torch.tensor(atomic_num, dtype=torch.long)
        

            atom_features = np.vstack([ari.get_atom_features(crystal[i].specie.number)
                               for i in range(len(crystal))])

            x = torch.tensor(atom_features)

            #all_neighbors, max_num_neighbors, edge_index, distances, edge_nn_vec = get_neighbors_crystalnn(crystal)
            #all_neighbors, max_num_neighbors, 
            #edge_index, distances, edge_vec = get_neighbors_crystalnn(crystal)
            #data_pro_dir = '/GPUFS/nscc-gz_pinchen2/apps/deepLearning/pytorch/matformer-equi/v10/datasets/mp/raw/pro_cut/'
            data_pro_dir = '/GPUFS/nscc-gz_pinchen2/apps/deepLearning/pytorch/matformer-equi/v10/datasets/mp/raw/pro_cut_knn/'
            #edge_index, distances, edge_vec = pickle.load(open(data_pro_dir + file_name + '.cif.p','rb'))
            #for test get_neighbor_graph.
            edge_src, edge_dst, distances, edge_vec = pickle.load(open(data_pro_dir + file_name + '.cif.p','rb'))
            #print(edge_vec)
            #exit()
            #r_cut = 5
            #max_neighbors = 12
            #lat = crystal.lattice
            #Seting self_interaction to True to include self edge.
            #edge_src, edge_dst, edge_distances, edge_vec = ase.neighborlist.neighbor_list("ijdD", a=crystal2, cutoff=r_cut, self_interaction=True)
            #min_nbr = min((natom) for natom in  np.bincount(edge_src))
            #if min_nbr < max_neighbors:
            #    lat = crystal.lattice
            #    if r_cut < max(lat.a, lat.b, lat.c):
            #        r_cut = max(lat.a, lat.b, lat.c) + 2 
            #    else:
            #        r_cut = 2 * r_cut + 2
            #    edge_src, edge_dst, edge_distances, edge_vec = ase.neighborlist.neighbor_list("ijdD", a=crystal2, cutoff=r_cut, self_interaction=True)

            #all_neighbors = [sorted(neighbors, key=lambda x: x[1]) for neighbors in all_neighbors]
            
            #max_neigh_index = np.array([])
            #max_neigh_index = []
	    ##Forcely cutting distances method.##
            #for i in range(len(crystal.sites)):
            #    idx_i = (edge_src == i).nonzero()[0]
            #    idx_sorted = np.argsort(edge_distances[idx_i])[: max_neighbors]
            #    max_neigh_index.append(idx_i[idx_sorted])
	    ##K-NN method.
            #for i in range(len(crystal.sites)):
            #    idx_i = (edge_src == i).nonzero()[0]
            #    #distance_sorted_index = np.argsort(edge_distances[idx_i])
            #    distance_sorted = np.sort(edge_distances[idx_i])
                #To include self edge, not using max_neighbors -1 ;
            #    max_dist = distance_sorted[max_neighbors-1]
            #    max_dist_index = np.where(edge_distances[idx_i] <= max_dist+0.001)
            #    max_dist_index = np.array(max_dist_index).flatten()

            #    max_neigh_index_t = [idx_i[i] for i in max_dist_index]
            #    max_neigh_index_t = np.array(max_neigh_index_t)
            #    max_neigh_index = np.append(max_neigh_index,max_neigh_index_t)

            #max_neigh_index=np.concatenate(max_neigh_index)
            #max_neigh_index=max_neigh_index.tolist()
            #max_neigh_index=np.array(max_neigh_index)
            #max_neigh_index=max_neigh_index.tolist()
            #max_neigh_index=max_neigh_index.flatten().astype(int)
            #max_neigh_index=max_neigh_index.tolist()
            
            #max_neigh_index=[max_neigh_index[i] for i in  range(len(max_neigh_index))]
            #print(max_neigh_index)
            #print(edge_src)
            #print(edge_src[max_neigh_index])

            
            #edge_src, edge_dst, edge_vec, distances=edge_src[max_neigh_index], edge_dst[max_neigh_index], edge_vec[max_neigh_index], edge_distances[max_neigh_index]
            #exit()
            #distances = gdf.expand(np.array(distances))
            distances = np.array(distances)
            #print(edge_src)
            #print(edge_dst)
            #print(distances)
            #exit()

            #edge_attr = torch.tensor(distances, dtype=torch.float)
            #edge_src = torch.tensor(edge_src, dtype=torch.long)
            #edge_dst = torch.tensor(edge_dst, dtype=torch.long)
            #edge_vec = torch.tensor(edge_vec, dtype=torch.float)
            #print(file_name)
            #exit()
	
            name = file_name
             
	    #build atom pairs within cutoff
            #edge_src, edge_dst, edge_vec, edge_distances = get_radius_graph(crystal,5,12)
            #print(edge_src)
            #edge_src, edge_dst = edge_index 
            edge_num = len(edge_src)
            edge_num = torch.tensor(edge_num,dtype=torch.long)
            edge_src = torch.tensor(edge_src,dtype=torch.long)
            edge_dst = torch.tensor(edge_dst,dtype=torch.long)
            #edge_vec = torch.tensor(list(edge_vec), dtype=torch.float)
            edge_vec = torch.tensor(edge_vec.astype(float), dtype=torch.float)
            edge_attr = torch.tensor(distances, dtype=torch.float)


            #print(edge_attr.shape)
            #print(edge_src)
            #print(edge_dst.shape)
            #print(edge_vec.shape)
            #exit()
            #data_list = []
            #print("addressning.")
            data = Data(x=x, z=z, edge_src=edge_src, edge_dst=edge_dst, 
                edge_attr=edge_attr, y=y, name=name, index=i,
                edge_vec = edge_vec, edge_num=edge_num)
                #edge_d_index=edge_d_index, edge_d_attr=edge_d_attr,
                #edge_src=edge_src, edge_dest=edge_dest, edge_vec=edge_vec, edge_distances=edge_distances)
            data_list.append(data)
            #print(data_list)
            #exit()
        torch.save(self.collate(data_list), self.processed_paths[0])





