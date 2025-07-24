## Note 
This project contains several GNN-based models for protein-ligand binding affinity prediction, which are reproduced from GIGN: https://github.com/guaguabujianle/GIGN

## Dataset
All data used in this paper are publicly available and can be accessed at https://zenodo.org/records/7490623#.Y60PTnZBxD8

If you want to download raw data, you can visit this website: 
https://www.pdbbind-plus.org.cn/download

## Requirements
Create a virtual environment and install the dependencies.
Install pytorch with the cuda version that fits your device
```
conda create -n HoloGNN python=3.8
conda activate HoloGNN
```
The packages you need to download are as follows
```
torch==1.12.0
torch_geometric==2.0.3
rdkit==2022.09.3  # important!!!
pymol==2.4.1
tqdm==4.63.0
```
We also provide a requirements.txt file to facilitate environment setup.
```
conda create –name HoloGNN –file requirements.txt 
```

## Usage
We provide a demo to show how to train, validate and test HoloGNN.
First, cd ./HoloGNN

### 1.Model training
Firstly, we need to download datasets and organize them as './data/train', 
'./data/valid', './data/test2013', './data/test2016', './data/test2019'.
Secondly, run python train_HoloGNN.py

### 2.affinity prediction for a single protein-ligand complex
we provide a script to predict affinity quickly from raw input data.
If you have already preformed docking between a protein and a ligand, 
you can use following script to predict affinity directly.
```
python predict_simpleone.py --protein_file protein.pdb --ligand_file mole.pdb
```

### 3.affinity prediction for a single protein with multiple ligands 
you can use run_pipeline.sh script to predict batch prediction for a single protein and multiple ligands:
```
bash run_pipeline.sh docking_dir
```

**Note:** 
First, place your folder inside the data directory, the folder structure should be like this:
```
data/
└── docking_dir/
    ├── mol1_1/
    │   ├── 6o0h_protein.pdb
    │   └── mol1_1.pdb
    ├── mol2_1/
    │   ├── 6o0h_protein.pdb
    │   └── mol2_1.pdb
...
```
The ligands files should be named molx_1.pdb file.
you can change the protein file path in preprocessing_docking.py, 
In this script, we use label.csv file to label molecules. In our experiment, we used 50 molecules, but 
you can add or remove entries based on the number of molecules you have —— just make sure to insert them in the file using the format molx_1, x.0.
Example format:
```
pdbid,-logKd/Ki
mol1_1,1.0
mol2_1,2.0
...
mol50_1,50.0
```