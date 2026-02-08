"""
Generate scenarios using Latin Hypercube Sampling (LHS)

Following the paper's specifications:
- 10,000 samples
- OD demand: 0-1,500 vehicles per OD pair
- Capacity: 4,000-26,000
- Speed: 45-80 km/h
"""

import numpy as np
from scipy.stats import qmc


def generate_lhs_samples(num_samples=10000, num_centroids=11, num_edges=76, seed=42):
    """
    Generate scenarios using Latin Hypercube Sampling.
    
    Args:
        num_samples (int): Number of scenarios to generate (default: 10,000)
        num_centroids (int): Number of centroid nodes (default: 11)
        num_edges (int): Number of edges in the network (default: 76)
        seed (int): Random seed for reproducibility
    
    Returns:
        od_matrices (np.ndarray): [num_samples, num_centroids, num_centroids]
        capacities (np.ndarray): [num_samples, num_edges]
        speeds (np.ndarray): [num_samples, num_edges]
    """
    print(f"\n{'='*60}")
    print("Generating scenarios with Latin Hypercube Sampling")
    print(f"{'='*60}")
    
    num_od_pairs = num_centroids * num_centroids  # 121
    num_dimensions = num_od_pairs + num_edges * 2  # 121 + 76 + 76 = 273
    
    print(f"  Total dimensions: {num_dimensions}")
    print(f"    - OD pairs: {num_od_pairs}")
    print(f"    - Capacities: {num_edges}")
    print(f"    - Speeds: {num_edges}")
    
    sampler = qmc.LatinHypercube(d=num_dimensions, seed=seed)
    
    print(f"\n  Sampling from {num_dimensions}-dimensional space...")
    samples = sampler.random(n=num_samples)  # [num_samples, num_dimensions]
    
    od_samples = samples[:, :num_od_pairs]
    capacity_samples = samples[:, num_od_pairs:num_od_pairs+num_edges]
    speed_samples = samples[:, num_od_pairs+num_edges:]
    
    print("  Scaling to actual ranges...")
    
    # OD demand: 0 - 1,500 vehicles
    od_matrices = od_samples * 1500.0
    od_matrices = od_matrices.reshape(num_samples, num_centroids, num_centroids)
    
    # Capacity: 4,000 - 26,000
    capacities = capacity_samples * (26000 - 4000) + 4000
    
    # Speed: 45 - 80 km/h
    speeds = speed_samples * (80 - 45) + 45
    
    print(f"\n  Generated data shapes:")
    print(f"    OD matrices: {od_matrices.shape}")
    print(f"    Capacities: {capacities.shape}")
    print(f"    Speeds: {speeds.shape}")
    
    print(f"\n  Data statistics:")
    print(f"    OD demand - min: {od_matrices.min():.2f}, max: {od_matrices.max():.2f}, mean: {od_matrices.mean():.2f}")
    print(f"    Capacity - min: {capacities.min():.2f}, max: {capacities.max():.2f}, mean: {capacities.mean():.2f}")
    print(f"    Speed - min: {speeds.min():.2f}, max: {speeds.max():.2f}, mean: {speeds.mean():.2f}")
    
    print(f"\n LHS sampling completed!")
    
    return od_matrices, capacities, speeds


def save_scenarios(od_matrices, capacities, speeds, save_path='processed_data/raw/scenarios.npz'):
    """
    Save generated scenarios to disk.
    
    Args:
        od_matrices: [num_samples, num_centroids, num_centroids]
        capacities: [num_samples, num_edges]
        speeds: [num_samples, num_edges]
        save_path: Path to save the .npz file
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez_compressed(
        save_path,
        od_matrices=od_matrices,
        capacities=capacities,
        speeds=speeds
    )
    print(f"\n Scenarios saved to: {save_path}")


def load_scenarios(load_path='processed_data/raw/scenarios.npz'):
    """
    Load previously generated scenarios.
    
    Returns:
        od_matrices, capacities, speeds
    """
    data = np.load(load_path)
    print(f"\n Scenarios loaded from: {load_path}")
    return data['od_matrices'], data['capacities'], data['speeds']

