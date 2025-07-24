import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import pickle
import argparse
from rdkit import Chem
import pymol
from rdkit import RDLogger

import pickle
import torch
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from HoloGNN import HoloGNN
from utils import load_model_dict

RDLogger.DisableLog('rdApp.*')

def generate_pocket(protein_path, ligand_path, output_pocket_path, distance=5):
    pymol.cmd.load(protein_path, 'protein')
    pymol.cmd.remove('resn HOH')
    pymol.cmd.load(ligand_path, 'ligand')
    pymol.cmd.remove('hydrogens')
    pymol.cmd.select('Pocket', f'byres ligand around {distance}')
    pymol.cmd.save(output_pocket_path, 'Pocket')
    pymol.cmd.delete('all')

def generate_complex(pocket_path, ligand_path, output_complex_path):
    ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    if ligand is None:
        print(f"❌ Unable to process ligand: {ligand_path}")
        return

    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    if pocket is None:
        print(f"❌ Unable to process pocket: {pocket_path}")
        return

    complex_data = (ligand, pocket)
    with open(output_complex_path, 'wb') as f:
        pickle.dump(complex_data, f)
    print(f"✅ Complex saved to {output_complex_path}")


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph):
    for atom in mol.GetAtoms():
        features = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']) + \
                   one_of_k_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5,6]) + \
                   one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) + \
                   one_of_k_encoding_unk(atom.GetHybridization(), [
                       Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                       Chem.rdchem.HybridizationType.SP3D2
                   ]) + [atom.GetIsAromatic()]
        features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4])
        graph.add_node(atom.GetIdx(), feats=torch.tensor(features, dtype=torch.float32))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter




def generate_pyg(complex_path):
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=1)
    edge_index_inter = inter_graph(ligand, pocket)

    pos_l = torch.tensor(ligand.GetConformers()[0].GetPositions(), dtype=torch.float32)
    pos_p = torch.tensor(pocket.GetConformers()[0].GetPositions(), dtype=torch.float32)
    pos = torch.cat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros(atom_num_l), torch.ones(atom_num_p)])

    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
                pos=pos, split=split)

    torch.save(data, 'complex_5A.pyg')
    print("✅ 保存完成：complex_5A.pyg")



def predict_affinity(pyg_file):


    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = './model/20250604_114310_HoloGNN_repeat1/model/epoch-314, train_loss-0.1701, train_rmse-0.4125, valid_rmse-1.1381, valid_pr-0.7888.pt'
    model = HoloGNN(35, 256).to(DEVICE)
    load_model_dict(model, MODEL_PATH)
    model.eval()
    data = torch.load(pyg_file).to(DEVICE)

    # ✅ 添加 batch 信息
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred = model(data).item()

    print(f'✅ Prediction for {pyg_file}: {pred:.4f}')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate protein pocket and complex')
    parser.add_argument('--protein_file', help='Path to the protein PDB file')
    parser.add_argument('--ligand_file', help='Path to the ligand file (pdb)')
    parser.add_argument('--distance', type=int, default=5, help='Distance (Å) to define pocket region (default: 5)')
    
    args = parser.parse_args()

    protein_path = args.protein_file
    ligand_path = args.ligand_file
    distance = args.distance

    pocket_path = f'Pocket_{distance}A.pdb'
    complex_path = f'complex_{distance}A.rdkit'

    print(f"🔁 Generating pocket within {distance}Å around ligand...")
    generate_pocket(protein_path, ligand_path, pocket_path, distance)

    print("🔁 Generating complex file...")
    generate_complex(pocket_path, ligand_path, complex_path)

    print("🔁 Generating pyg file...")  
    generate_pyg(complex_path)


    print("🔁 predict affinity...")  
    PYG_FILE = 'complex_5A.pyg'
    predict_affinity(PYG_FILE)

    print("✅ All steps completed.")
