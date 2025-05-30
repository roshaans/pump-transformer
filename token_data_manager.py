import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import h5py
import h5py
from typing import Dict, List, Tuple

class TokenDataManager:
    def __init__(self, base_dir: str):
        """
        Initialize TokenDataManager
        Args:
            base_dir: Base directory for storing token data
        """
        self.base_dir = base_dir
        self.sequences_dir = os.path.join(base_dir, 'sequences')
        self.metadata_file = os.path.join(base_dir, 'token_metadata.json')
        
        # Create directories if they don't exist
        os.makedirs(self.sequences_dir, exist_ok=True)
        
        # Load or create token metadata
        self.token_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load token metadata from disk or create new"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'token_ids': {}, 'next_token_id': 0}

    def _save_metadata(self):
        """Save token metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.token_metadata, f)

    def get_token_id(self, token_address: str) -> int:
        """Get numeric ID for token address, creating new if needed"""
        if token_address not in self.token_metadata['token_ids']:
            token_id = self.token_metadata['next_token_id']
            self.token_metadata['token_ids'][token_address] = token_id
            self.token_metadata['next_token_id'] = token_id + 1
            self._save_metadata()
        return self.token_metadata['token_ids'][token_address]

    def save_token_sequences(self, token_address: str, sequences: np.ndarray):
        """
        Save token sequences to disk using HDF5
        Args:
            token_address: Token contract address
            sequences: numpy array of shape (n_sequences, seq_length, n_features)
        """
        token_id = self.get_token_id(token_address)
        file_path = os.path.join(self.sequences_dir, f'token_{token_id}.h5')
        
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('sequences', data=sequences)

    def load_token_sequences(self, token_address: str) -> np.ndarray:
        """Load token sequences from disk"""
        token_id = self.get_token_id(token_address)
        file_path = os.path.join(self.sequences_dir, f'token_{token_id}.h5')
        
        with h5py.File(file_path, 'r') as f:
            return f['sequences'][:]

class MultiTokenDataset(Dataset):
    def __init__(self, token_manager: TokenDataManager, token_addresses: List[str], 
                 seq_length: int, forward_steps: int):
        self.token_manager = token_manager
        self.token_addresses = token_addresses
        self.seq_length = seq_length
        self.forward_steps = forward_steps
        
        # Load all sequences
        self.sequences = []
        self.token_ids = []
        
        for addr in token_addresses:
            token_seqs = token_manager.load_token_sequences(addr)
            token_id = token_manager.get_token_id(addr)
            
            # Create input/target pairs
            n_seqs = len(token_seqs) - seq_length - forward_steps + 1
            for i in range(n_seqs):
                self.sequences.append(token_seqs[i:i+seq_length+forward_steps])
                self.token_ids.append(token_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_id = self.token_ids[idx]
        
        X = torch.FloatTensor(seq[:self.seq_length])
        y = torch.FloatTensor(seq[self.seq_length:, 0])  # Only predict price
        token_id = torch.LongTensor([token_id])
        
        return X, token_id, y
