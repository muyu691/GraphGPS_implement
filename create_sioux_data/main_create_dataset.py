"""
Main script to create the complete Sioux Falls Traffic Assignment dataset.

This script orchestrates all steps:
1. Load Sioux Falls network
2. Generate scenarios using LHS
3. Solve SUE for all scenarios
4. Build PyG dataset
5. Normalize data
6. Split into train/val/test
7. Generate OOD data
8. Save everything

Usage:
    python main_create_dataset.py --num_samples 10000 --method frank_wolfe
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_sioux import load_sioux_falls_network
from generate_scenarios import generate_lhs_samples, save_scenarios, load_scenarios
from sue_solver import solve_sue_batch, save_flows, load_flows
from build_pyg_data import build_dataset, save_dataset, load_dataset
from preprocess import normalize_dataset, split_dataset, save_scalers
from generate_ood import generate_all_ood_variants, save_ood_dataset
from utils import (create_data_directories, validate_data_shapes, 
                   check_for_nans_and_infs, compute_statistics)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create Sioux Falls Traffic Assignment Dataset'
    )
    
    # Main parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of scenarios to generate (default: 10000)')
    parser.add_argument('--sue_method', type=str, default='frank_wolfe',
                       choices=['frank_wolfe'],
                       help='SUE solver method (default: frank_wolfe)')
    
    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set ratio (default: 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    
    # OOD generation
    parser.add_argument('--generate_ood', action='store_true',
                       help='Generate OOD datasets')
    parser.add_argument('--ood_perturbation_levels', nargs='+', type=int,
                       default=[10, 50, 90],
                       help='OOD perturbation levels (default: 10 50 90)')
    
    # Paths
    parser.add_argument('--network_file', type=str,
                       default='../sioux_data/SiouxFalls_net.tntp',
                       help='Path to Sioux Falls network file')
    parser.add_argument('--output_dir', type=str,
                       default='processed_data',
                       help='Output directory for processed data')
    
    # Options
    parser.add_argument('--skip_scenarios', action='store_true',
                       help='Skip scenario generation (use existing)')
    parser.add_argument('--skip_sue', action='store_true',
                       help='Skip SUE solving (use existing flows)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "SIOUX FALLS DATASET CREATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Create directories
    print(f"\n{'='*70}")
    print("STEP 0: Setting up directories")
    print(f"{'='*70}")
    create_data_directories()
    
    
    # STEP 1: Load Sioux Falls Network
    print(f"\n{'='*70}")
    print("STEP 1: Loading Sioux Falls Network")
    print(f"{'='*70}")
    
    G, centroids = load_sioux_falls_network(args.network_file)
    
    # STEP 2: Generate or Load Scenarios
    print(f"\n{'='*70}")
    print("STEP 2: Generating/Loading Scenarios")
    print(f"{'='*70}")
    
    scenarios_path = f"{args.output_dir}/raw/scenarios.npz"
    
    if args.skip_scenarios and os.path.exists(scenarios_path):
        print("  Loading existing scenarios...")
        od_matrices, capacities, speeds = load_scenarios(scenarios_path)
    else:
        print("  Generating new scenarios with LHS...")
        od_matrices, capacities, speeds = generate_lhs_samples(
            num_samples=args.num_samples,
            num_centroids=11,
            num_edges=76,
            seed=args.seed
        )
        save_scenarios(od_matrices, capacities, speeds, scenarios_path)
    
    # Validate
    validate_data_shapes(od_matrices, capacities, speeds)
    
    # STEP 3: Solve SUE or Load Flows
    print(f"\n{'='*70}")
    print("STEP 3: Solving SUE Traffic Assignment")
    print(f"{'='*70}")
    
    flows_path = f"{args.output_dir}/raw/flows.npz"
    
    if args.skip_sue and os.path.exists(flows_path):
        print("  Loading existing flows...")
        flows = load_flows(flows_path)
    else:
        print("  Running SUE assignment...")
        print(f"Warning: This may take several hours for {args.num_samples} samples!")
        flows = solve_sue_batch(
            G, od_matrices, capacities, speeds,
            method=args.sue_method,
            verbose=args.verbose
        )
        save_flows(flows, flows_path)
    
    # Validate
    validate_data_shapes(od_matrices, capacities, speeds, flows)
    
    # Check for issues
    check_for_nans_and_infs({
        'OD matrices': od_matrices,
        'Capacities': capacities,
        'Speeds': speeds,
        'Flows': flows
    })
    
    # STEP 4: Build PyG Dataset
    print(f"\n{'='*70}")
    print("STEP 4: Building PyTorch Geometric Dataset")
    print(f"{'='*70}")
    
    dataset = build_dataset(
        G, od_matrices, capacities, speeds, flows, centroids,
        verbose=args.verbose
    )
    
    # Save raw dataset
    raw_dataset_path = f"{args.output_dir}/raw/dataset_raw.pt"
    save_dataset(dataset, raw_dataset_path)
    
    # STEP 5: Normalize Data
    print(f"\n{'='*70}")
    print("STEP 5: Normalizing Data (MinMax)")
    print(f"{'='*70}")
    
    normalized_dataset, scalers = normalize_dataset(dataset)
    
    # Save scalers
    scalers_path = f"{args.output_dir}/processed/scalers.pkl"
    save_scalers(scalers, scalers_path)
    
    # STEP 6: Split Dataset
    print(f"\n{'='*70}")
    print("STEP 6: Splitting Dataset")
    print(f"{'='*70}")
    
    train_dataset, val_dataset, test_dataset = split_dataset(
        normalized_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save splits
    train_path = f"{args.output_dir}/processed/train_dataset.pt"
    val_path = f"{args.output_dir}/processed/val_dataset.pt"
    test_path = f"{args.output_dir}/processed/test_dataset.pt"
    
    save_dataset(train_dataset, train_path)
    save_dataset(val_dataset, val_path)
    save_dataset(test_dataset, test_path)
    
    # # STEP 7: Generate OOD Data (Optional)
    # if args.generate_ood:
    #     print(f"\n{'='*70}")
    #     print("STEP 7: Generating OOD Datasets")
    #     print(f"{'='*70}")
        
    #     ood_datasets = generate_all_ood_variants(
    #         G, normalized_dataset, scalers,
    #         perturbation_levels=args.ood_perturbation_levels
    #     )
        
    #     # Save OOD datasets
    #     for name, ood_data in ood_datasets.items():
    #         ood_path = f"{args.output_dir}/ood/{name}.pt"
    #         save_ood_dataset(ood_data, ood_path)
    
    # STEP 8: Final Summary
    print(f"\n{'='*70}")
    print("DATASET CREATION COMPLETED!")
    print(f"{'='*70}")
    
    print(f"\n Summary:")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # if args.generate_ood:
    #     print(f"  OOD datasets: {len(ood_datasets)}")
    
    print(f"\n Output files:")
    print(f"  Raw scenarios: {scenarios_path}")
    print(f"  Flows: {flows_path}")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    print(f"  Scalers: {scalers_path}")
    
    # if args.generate_ood:
    #     print(f"  OOD: {args.output_dir}/ood/")
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
