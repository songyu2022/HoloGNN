# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %%
def generate_pocket(data_dir, distance=5):
    # 获取当前目录下的所有子目录
    print(data_dir)
    all_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(all_folders)

    # 筛选出只包含 molX_1 格式的文件夹
    data_dir_1 = [d for d in all_folders if d.startswith('mol') and d.endswith('_1')]
    print(data_dir_1)

    # 遍历目标文件夹
    for cid in sorted(data_dir_1):
        print(f"处理目录: {cid}")
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}.pdb")
        protein_path= os.path.join(complex_dir, "6o0h_protein.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid} around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, cid, f'{cid}.pdb')

        save_path = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data preprocess ")
    parser.add_argument('--docking_dir', type=str, required=True, help='the result of docking')
    parser.add_argument('--csv_file', type=str, required=True, help='the index of molecule')
    parser.add_argument('--distance', type=int, default=5, help='distance')
    parser.add_argument('--ligand_format', type=str, default='pdb', help='the form of ligand')
    args = parser.parse_args()

    docking_dir = args.docking_dir
    csv_file = args.csv_file
    distance = args.distance
    input_ligand_format = args.ligand_format
    # distance = 5
    # input_ligand_format = 'pdb'
    data_root = './data'
    data_dir = os.path.join(data_root, docking_dir)
    data_df = pd.read_csv(os.path.join(data_root, csv_file))

    ## generate pocket within 5 Ångström around ligand 
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)



# %%
