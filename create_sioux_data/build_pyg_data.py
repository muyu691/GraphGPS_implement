"""
Build PyTorch Geometric Data objects from Sioux Falls scenarios.

Follows the paper's structure:
- Node features: OD matrix rows for centroids, zeros for non-centroids [24, 11]
- Edge features: [free_flow_time, speed, capacity] [76, 3]
- Labels: Edge flows [76]
"""

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def create_node_features(od_matrix, centroids, num_nodes=24):
    """
    Create node feature matrix following the paper's specification.
    
    Args:
        od_matrix: [num_centroids, num_centroids] OD matrix
        centroids: List of centroid node IDs (e.g., [1,2,...,11])
        num_nodes: Total number of nodes (24 for Sioux Falls)
    
    Returns:
        node_features: [num_nodes, num_centroids] node feature matrix
    """
    num_centroids = len(centroids)
    node_features = np.zeros((num_nodes, num_centroids), dtype=np.float32)
    
    # Fill features for centroid nodes with their OD matrix row
    for i, centroid_id in enumerate(centroids):
        # Convert 1-indexed node ID to 0-indexed array index
        node_idx = centroid_id - 1
        node_features[node_idx] = od_matrix[i, :]
    
    # Non-centroid nodes already have zero features
    
    return node_features


def create_edge_features(G, free_flow_times, speeds, capacities):
    """
    Create edge feature matrix: [free_flow_time, speed, capacity]
    
    Args:
        G: NetworkX graph
        free_flow_times: [num_edges] free-flow times
        speeds: [num_edges] speed limits
        capacities: [num_edges] link capacities
    
    Returns:
        edge_features: [num_edges, 3] edge feature matrix
        edge_index: [2, num_edges] PyG-format edge connectivity
    """
    edges = list(G.edges())
    num_edges = len(edges)
    
    # Stack edge features: [free_flow_time, speed, capacity]
    edge_features = np.stack([
        free_flow_times,
        speeds,
        capacities
    ], axis=1).astype(np.float32)  # [num_edges, 3]
    
    # Create edge_index in PyG format [2, num_edges]
    # Convert from 1-indexed to 0-indexed
    edge_index = np.array([[u-1, v-1] for u, v in edges], dtype=np.int64).T
    
    return edge_features, edge_index


def create_pyg_data(node_features, edge_index, edge_features, flows):
    """
    Create a PyTorch Geometric Data object.
    
    Args:
        node_features: [num_nodes, num_centroids] numpy array
        edge_index: [2, num_edges] numpy array
        edge_features: [num_edges, 3] numpy array
        flows: [num_edges] numpy array (labels)
    
    Returns:
        data: PyG Data object
    """
    data = Data(
        x=torch.FloatTensor(node_features),       # [24, 11]
        edge_index=torch.LongTensor(edge_index),  # [2, 76]
        edge_attr=torch.FloatTensor(edge_features), # [76, 3]
        y=torch.FloatTensor(flows)                # [76]
    )
    
    return data


def build_dataset(G, od_matrices, capacities, speeds, flows, centroids, verbose=True):
    """
    Build a complete PyG dataset from scenarios.
    
    Args:
        G: NetworkX graph
        od_matrices: [num_samples, num_centroids, num_centroids]
        capacities: [num_samples, num_edges]
        speeds: [num_samples, num_edges]
        flows: [num_samples, num_edges]
        centroids: List of centroid node IDs
        verbose: Show progress bar
    
    Returns:
        dataset: List of PyG Data objects
    """
    try:
        from .utils import compute_free_flow_times
    except ImportError:
        from utils import compute_free_flow_times
    
    num_samples = od_matrices.shape[0]
    num_nodes = G.number_of_nodes()
    
    print(f"\n{'='*60}")
    print(f"Building PyTorch Geometric dataset")
    print(f"{'='*60}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {len(list(G.edges()))}")
    print(f"  Number of centroids: {len(centroids)}")
    
    # Compute free-flow times for all scenarios
    print("  Computing free-flow times...")
    free_flow_times = compute_free_flow_times(G, speeds)
    
    # Get edge_index (same for all samples)
    edges = list(G.edges())
    edge_index = np.array([[u-1, v-1] for u, v in edges], dtype=np.int64).T
    
    dataset = []
    
    print("  Creating Data objects...")
    iterator = tqdm(range(num_samples), desc="  Progress") if verbose else range(num_samples)
    
    for i in iterator:
        # Create node features
        node_features = create_node_features(
            od_matrices[i],
            centroids,
            num_nodes
        )
        
        # Create edge features
        edge_features = np.stack([
            free_flow_times[i],
            speeds[i],
            capacities[i]
        ], axis=1).astype(np.float32)
        
        # Create PyG Data object
        data = create_pyg_data(
            node_features,
            edge_index,
            edge_features,
            flows[i]
        )
        
        dataset.append(data)
    
    print(f"\n Dataset created with {len(dataset)} samples!")
    
    # Print example
    print(f"\n  Example Data object:")
    print(f"    x (node features): {dataset[0].x.shape}")
    print(f"    edge_index: {dataset[0].edge_index.shape}")
    print(f"    edge_attr (edge features): {dataset[0].edge_attr.shape}")
    print(f"    y (flow labels): {dataset[0].y.shape}")
    
    return dataset


def save_dataset(dataset, save_path):
    """
    Save PyG dataset to disk.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(dataset, save_path)
    print(f"\n Dataset saved to: {save_path}")


def load_dataset(load_path):
    """
    Load PyG dataset from disk.
    """
    dataset = torch.load(load_path)
    print(f"\n Dataset loaded from: {load_path}")
    print(f"  Number of samples: {len(dataset)}")
    return dataset
