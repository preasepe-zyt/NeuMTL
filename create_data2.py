import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import re
from typing import List
import string
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms
import torch
from PIL import Image




def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]
    
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + #Atom symbol
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Number of adjacent atoms
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # Number of adjacent hydrogens
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Implicit valence
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + #Formal charge
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + #Hybridization
                    [atom.GetIsAromatic()] + #Aromaticity
                    [atom.IsInRing()] #In ring
                    )

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(bond_feats)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

   
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edge_feats = bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'edge_feats': edge_feats}))
        
    g = nx.Graph()
    g.add_edges_from(edges)
    g = g.to_directed()
    edge_index = []
    edge_feats = []
    for e1, e2, feats in g.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(feats['edge_feats'])

    
    return c_size, features, edge_index, edge_feats




def smile_parse(smiles, tokenizer: Tokenizer):
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    smi = tokenizer.parse(smiles)
    return smi


def smiles_to_sequence(smiles, max_len=138, charset=None):
    if charset is None:
        charset = list(string.ascii_letters + string.digits + "()-=#$@+/\\")  # 你可根据需要扩展字符集
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}  # 0留给padding
    sequence = [char_to_idx.get(char, 0) for char in smiles]
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return np.array(sequence, dtype=np.int32)


def smiles_to_img(smile, img_size=112):
    """
    将单个 SMILES 转为图像张量。
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    else:
        img = Draw.MolToImage(mol, size=(img_size, img_size))
    
    tensor = transform(img)
    return tensor
    

Smiles = []
for dt_name in ['BBB',"to_NA","to_NT","to_NC"]: #
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data2/' + dt_name + '_' + opt + '.csv')
        Smiles += list( df['Smiles'] )
Smiles = set(Smiles)


smile_graph = {}
smile_sequence = {}
smile_img = {}
for smile in Smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

    se = smiles_to_sequence(smile)
    smile_sequence[smile] = se

    img = smiles_to_img(smile)
    smile_img[smile] = img


    
dir = 'data2'
datasets = ['BBB',"to_NA","to_NT","to_NC"]
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data2/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data2/processed/' + dataset + '_test.pt'
    tokenizer_file = f'{dir}/{dataset}_tokenizer.pkl'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df_train = pd.read_csv('data2/' + dataset + '_train.csv')
        df_test = pd.read_csv('data2/' + dataset + '_test.csv')

        all_smiles = set(df_train['Smiles']).union(set(df_test['Smiles']))
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))

        with open(tokenizer_file, 'wb') as file:
            pickle.dump(tokenizer, file)        
        # Process train set
        train_drugs, train_Y = list(df_train['Smiles']), list(df_train['Label'])
 
        train_drugs, train_Y = np.asarray(train_drugs), np.asarray(train_Y)


        # Process test set
        test_drugs, test_Y = list(df_test['Smiles']), list(df_test['Label'])
 
        test_drugs,  test_Y = np.asarray(test_drugs), np.asarray(test_Y)        
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset_drug(root='data2', dataset=dataset+'_train', xd=train_drugs, y=train_Y, smile_graph=smile_graph, smile_sequences=smile_sequence, smile_imgs=smile_img)
        
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset_drug(root='data2', dataset=dataset+'_test', xd=test_drugs, y=test_Y, smile_graph=smile_graph, smile_sequences=smile_sequence, smile_imgs=smile_img)
        
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
