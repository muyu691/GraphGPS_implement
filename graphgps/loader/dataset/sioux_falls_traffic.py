"""
Sioux Falls Traffic Flow Dataset for GraphGPS

This dataset loader loads the preprocessed Sioux Falls traffic assignment data
created by create_sioux_data/main_create_dataset.py
"""

import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset


class SiouxFallsTrafficDataset(InMemoryDataset):
    """
    Sioux Falls Traffic Flow Prediction Dataset
    
    Task: Predict equilibrium traffic flow on each edge given:
        - OD demand matrix (node features)
        - Network properties (edge features: capacity, speed, free-flow time)
    
    Dataset statistics:
        - 24 nodes (intersections)
        - 76 directed edges (road segments)
        - 11 centroids (traffic zones)
        - 10,000 scenarios (default)
    
    Node features (x): [24, 11]
        - Rows 0-10 (centroids): OD demand distribution
        - Rows 11-23 (non-centroids): zeros
    
    Edge features (edge_attr): [76, 3]
        - Free-flow time (minutes)
        - Speed limit (km/h)
        - Capacity (vehicles/hour)
    
    Labels (y): [76,]
        - Equilibrium flow on each edge (vehicles/hour)
    """
    
    def __init__(self, root='datasets/sioux_falls', 
                 split='train',
                 transform=None, 
                 pre_transform=None,
                 pre_filter=None):
        """
        Args:
            root: Root directory where the dataset is stored
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to each data object
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
        """
        self.split = split
        assert split in ['train', 'val', 'test'], \
            f"Split must be 'train', 'val', or 'test', got {split}"
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load the appropriate split
        # The saved files are lists of Data objects, need to collate them
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:  # test
            path = self.processed_paths[2]
        
        # Load the list of Data objects
        data_list = torch.load(path)
        
        # Collate into InMemoryDataset format
        self.data, self.slices = self.collate(data_list)
    
    @property
    def raw_file_names(self):
        """Files needed in raw_dir"""
        return []  # We use already processed data
    
    @property
    def processed_dir(self):
        """Override to use the root directory directly as processed dir"""
        return osp.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        """Expected processed files"""
        return ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']
    
    def download(self):
        """
        No download needed - data should be created by
        create_sioux_data/main_create_dataset.py
        """
        if not osp.exists(self.processed_dir):
            raise FileNotFoundError(
                f"Processed data not found in {self.processed_dir}.\n"
                "Please run: cd create_sioux_data && "
                "python main_create_dataset.py --num_samples 10000"
            )
    
    def process(self):
        """
        No processing needed - we directly use the .pt files created by
        main_create_dataset.py
        """
        pass
    
    def get_idx_split(self):
        """
        Return train/val/test indices.
        Since we have separate files, return simple ranges.
        """
        # Get the number of samples in each split
        if hasattr(self, 'data'):
            n_samples = self.slices['x'].shape[0] - 1
        else:
            # Load to check size
            data, slices = torch.load(self.processed_paths[0])
            n_train = slices['x'].shape[0] - 1
            data, slices = torch.load(self.processed_paths[1])
            n_val = slices['x'].shape[0] - 1
            data, slices = torch.load(self.processed_paths[2])
            n_test = slices['x'].shape[0] - 1
            
            return {
                'train': torch.arange(n_train),
                'valid': torch.arange(n_val),
                'test': torch.arange(n_test)
            }
        
        return {
            'train': torch.arange(n_samples),
            'valid': torch.arange(n_samples),
            'test': torch.arange(n_samples)
        }
    
    def __repr__(self):
        return f'{self.__class__.__name__}(split={self.split}, ' \
               f'num_graphs={len(self)})'
