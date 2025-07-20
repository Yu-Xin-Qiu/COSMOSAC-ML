from __future__ import print_function, division
import functools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):
        self.smiles = df['Normalized SMILES']
        self.tokens = np.array(
            [tokenizer.encode(i, max_length=100, truncation=True, padding='max_length') 
            for i in self.smiles]
        )


        self.label = df['Vcosmo']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Tokenize the SMILES string to get X (shape should be [100, 1] if padding length is 100)
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        
        y = torch.from_numpy(np.asarray(self.label[index])).float() 

        smiles = self.smiles[index]

        return (X, y), y, smiles

class SMILES_sigma_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):
        self.smiles = df['Normalized SMILES']
        self.tokens = np.array(
            [tokenizer.encode(i, max_length=100, truncation=True, padding='max_length') 
            for i in self.smiles]
        )

        self.label = df[[f'sigma_{i}' for i in range(1, 52)]].values  # Ensure it's a numpy array
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Tokenize the SMILES string to get X (shape should be [100, 1] if padding length is 100)
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))

        # Ensure y has shape (51,) where each value corresponds to a sigma_i
        y = torch.from_numpy(np.asarray(self.label[index])).float()  # Shape should be [51]

        smiles = self.smiles[index]
        # print(y)
        # print(X.shape, y.shape, smiles)

        return (X, y),y , smiles


class SMILES_dataset_IDAC(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):

        self.smiles_x = df['SMILES_solute']
        self.smiles_y = df['SMILES_solvent']

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens_x = np.array(
            list(Parallel(n_jobs=3)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles_x, desc='Tokenizing SMILES_solute', total=len(self.smiles_x))
            ))
        )
        self.tokens_y = np.array(
            list(Parallel(n_jobs=3)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles_y, desc='Tokenizing SMILES_solvent', total=len(self.smiles_y))
            ))
        )

        self.tokenizer = tokenizer
        self.t = df['T/K']
        self.AC = df['lngamma1']
        self.AC_COSMO = df['COSMO-SAC1']
        self.label = df['dlnr1']

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens_x[index]).astype(np.float32))
        Y = torch.from_numpy(np.asarray(self.tokens_y[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).float()
        smiles_x = self.smiles_x[index]
        smiles_y = self.smiles_y[index]
        t = torch.from_numpy(np.asarray(self.t[index])).view(-1, 1)
        AC = torch.from_numpy(np.asarray(self.AC[index])).view(-1, 1)
        AC_COSMO = torch.from_numpy(np.asarray(self.AC_COSMO[index])).view(-1, 1)

        return (X, Y, t), y.float(), smiles_x, smiles_y, t,AC,AC_COSMO