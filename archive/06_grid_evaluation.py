"""
Grid Evaluation for SBM Hyperparameters

This script performs a grid search over SBM hyperparameters and evaluates
the robustness of mutation rankings across different sub-alignments.

@author: Generated for Shoichi's PhD project
"""

####################### MODULES #######################
import numpy as np
import subprocess
import itertools
from pathlib import Path
from glob import glob
import os
from scipy.stats import kendalltau
import csv

# Import local modules
import utils_COMBINE as uc
from config import (
    POS_ALIGN_Yip_RedSec as ALIGN_MAR_RED_23,
    SEQ_3TGI_YIP_RedSec as SEQ_REF_REDSEC_23,
    Ind_RedSec_Sho as MARION_RED_SECTOR_23,
    POS_ALIGN_Yip,
    SEQ_3TGI_YIP,
)
import utils as ut

#######################################################

####################### CONFIGURATION #######################

# Parameter grid (~48 combinations)
PARAM_GRID = {
    'N_chains': [150, 250, 350],
    'N_iter': [500, 800],
    'k_MCMC': [5000, 10000],
    'theta': [0.15, 0.2],
    'lambda_J': [0, 0.01],
}

# Fixed parameters
FIXED_PARAMS = {
    'TestTrain': 0,
    'm': 1,
    'rep': 1,
    'N_av': 1,
    'ParamInit': 'zero',
    'lambdh': 0,
}

# Sub-alignments to use (5 out of 10)
SUBALN_INDICES = [0, 1, 2, 3, 4]

# Paths - Use RedSecMar (Marion's 23-residue sector definition)
DATA_DIR = Path('./data/RedSecMar')
RESULTS_DIR = Path('./results')
GRID_RESULTS_DIR = RESULTS_DIR / 'grid_evaluation'

# Family name prefix (to match existing conventions from 04_train_subaln_sbm.sh)
FAMILY_PREFIX = "GridEval"  # Results will be in results/GridEval_SubAln_X/

# Mutation prediction settings
NB_MUTATIONS = 10  # Number of mutations to predict for ranking

# Sector definitions for different lengths
# 22-residue sector (Shoichi's original)
RED_SECTOR_22 = sorted([197, 239, 237, 224, 186, 225, 189, 190, 200, 227, 228, 222, 238, 2, 229, 164, 195, 194, 231, 165, 176, 1])
ALIGN_RED_22 = POS_ALIGN_Yip[RED_SECTOR_22]
SEQ_REF_22 = "".join([SEQ_3TGI_YIP[i] for i in RED_SECTOR_22])

# 23-residue sector (Marion's definition) - imported from config
ALIGN_RED_23 = ALIGN_MAR_RED_23
SEQ_REF_23 = SEQ_REF_REDSEC_23

#######################################################

####################### TRAINING FUNCTIONS #######################

def get_param_combinations():
    """Generate all parameter combinations from the grid."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def get_model_filename(family_name, params):
    """Construct the expected model filename based on parameters."""
    filename = f"{family_name}_ModelSBM"
    filename += f"_N_chains{params['N_chains']}"
    filename += f"_N_iter{params['N_iter']}"
    filename += f"_Param_init{FIXED_PARAMS['ParamInit']}"
    filename += f"_k_MCMC{params['k_MCMC']}"
    filename += f"_lambda_J{params['lambda_J']}"
    filename += f"_lambda_h{FIXED_PARAMS['lambdh']}"
    filename += f"_m{FIXED_PARAMS['m']}"
    filename += f"_theta{params['theta']}"
    filename += f"_N_Av{FIXED_PARAMS['N_av']}"
    filename += "_R0.npy"
    return filename


def train_single_model(subaln_idx, params, skip_if_exists=True):
    """
    Train a single SBM model with given parameters on a specific sub-alignment.
    
    Args:
        subaln_idx: Index of the sub-alignment (0-9)
        params: Dictionary of hyperparameters
        skip_if_exists: If True, skip training if model file already exists
    
    Returns:
        Path to the trained model file
    """
    family_name = f"GridEval_SubAln_{subaln_idx}"
    input_msa = DATA_DIR / f"subaln_seq_{subaln_idx}.npy"
    
    # Check if model already exists
    model_dir = RESULTS_DIR / family_name
    expected_filename = get_model_filename(family_name, params)
    expected_path = model_dir / expected_filename
    
    if skip_if_exists and expected_path.exists():
        print(f"  [SKIP] Model already exists: {expected_path.name}")
        return expected_path
    
    # Construct command
    cmd = [
        'python', 'SBM-CM-family.py',
        family_name,
        str(input_msa),
        '--TestTrain', str(FIXED_PARAMS['TestTrain']),
        '--m', str(FIXED_PARAMS['m']),
        '--k_MCMC', str(params['k_MCMC']),
        '--rep', str(FIXED_PARAMS['rep']),
        '--N_av', str(FIXED_PARAMS['N_av']),
        '--N_iter', str(params['N_iter']),
        '--theta', str(params['theta']),
        '--ParamInit', FIXED_PARAMS['ParamInit'],
        '--lambdJ', str(params['lambda_J']),
        '--lambdh', str(FIXED_PARAMS['lambdh']),
        '--N_chains', str(params['N_chains']),
    ]
    
    print(f"  [TRAIN] SubAln {subaln_idx}: N_chains={params['N_chains']}, N_iter={params['N_iter']}, "
          f"k_MCMC={params['k_MCMC']}, theta={params['theta']}, lambda_J={params['lambda_J']}")
    
    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for SubAln {subaln_idx}")
        print(f"  STDERR: {result.stderr[:500]}")
        return None
    
    # Find the generated model file
    if expected_path.exists():
        return expected_path
    
    # If exact path doesn't exist, search for it
    pattern = f"{family_name}_ModelSBM_N_chains{params['N_chains']}_N_iter{params['N_iter']}*.npy"
    matches = list(model_dir.glob(pattern))
    if matches:
        return matches[0]
    
    print(f"  [ERROR] Could not find trained model for SubAln {subaln_idx}")
    return None

#######################################################

####################### MUTATION RANKING #######################

def extract_mutation_ranking(model_path, nb_mutations=NB_MUTATIONS):
    """
    Extract mutation ranking from a trained model using DDE2 method.
    Auto-detects model sequence length and uses matching definitions.
    
    Args:
        model_path: Path to the trained model .npy file
        nb_mutations: Number of mutations to predict
    
    Returns:
        List of (amino_acid_idx, position_idx) tuples representing ranked mutations,
        or None if extraction fails
    """
    try:
        # Load model to detect sequence length
        output = np.load(str(model_path), allow_pickle=True)[()]
        h = output['h']
        model_length = h.shape[0]
        
        # Select appropriate definitions based on model length
        if model_length == 23:
            pos_align = ALIGN_RED_23
            seq_ref = SEQ_REF_23
        elif model_length == 22:
            pos_align = ALIGN_RED_22
            seq_ref = SEQ_REF_22
        else:
            print(f"  [ERROR] Unexpected model length: {model_length}")
            return None
        
        # Suppress print output from Propose_Mutation_DDE2
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            mutations = uc.Propose_Mutation_DDE2(
                str(model_path),
                nb_mutations,
                [('S', '189')],  # Initial mutation (D189S)
                POS_ALIGN=pos_align,
                SEQ_REF=seq_ref
            )
        finally:
            sys.stdout = old_stdout
        
        return mutations
    except Exception as e:
        print(f"  [ERROR] Failed to extract mutations from {model_path}: {e}")
        return None

#######################################################

####################### ROBUSTNESS METRICS #######################

def compute_kendall_tau_matrix(rankings_list):
    """
    Compute pairwise Kendall's tau between all ranking pairs.
    
    Args:
        rankings_list: List of rankings (each ranking is a list of (aa, pos) tuples)
    
    Returns:
        Mean Kendall's tau across all pairs, and the full matrix
    """
    n = len(rankings_list)
    if n < 2:
        return 0.0, np.array([[1.0]])
    
    tau_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                tau_matrix[i, j] = 1.0
            elif i < j:
                # Convert rankings to comparable format
                # Use the index in the ranking as the rank value
                ranking_i = rankings_list[i]
                ranking_j = rankings_list[j]
                
                if ranking_i is None or ranking_j is None:
                    tau_matrix[i, j] = np.nan
                    tau_matrix[j, i] = np.nan
                    continue
                
                # Create a common set of mutations
                set_i = set(ranking_i)
                set_j = set(ranking_j)
                common = set_i & set_j
                
                if len(common) < 2:
                    tau_matrix[i, j] = np.nan
                    tau_matrix[j, i] = np.nan
                    continue
                
                # Get ranks for common mutations
                ranks_i = []
                ranks_j = []
                for mut in common:
                    ranks_i.append(ranking_i.index(mut))
                    ranks_j.append(ranking_j.index(mut))
                
                tau, _ = kendalltau(ranks_i, ranks_j)
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau
    
    # Compute mean of upper triangle (excluding diagonal)
    upper_tri = tau_matrix[np.triu_indices(n, k=1)]
    valid_taus = upper_tri[~np.isnan(upper_tri)]
    
    mean_tau = np.nanmean(valid_taus) if len(valid_taus) > 0 else np.nan
    
    return mean_tau, tau_matrix


def compute_topk_jaccard(rankings_list, k_values=[3, 5, 10]):
    """
    Compute top-k Jaccard similarity across rankings.
    
    Args:
        rankings_list: List of rankings
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary mapping k to Jaccard similarity score
    """
    results = {}
    
    valid_rankings = [r for r in rankings_list if r is not None]
    if len(valid_rankings) < 2:
        return {k: np.nan for k in k_values}
    
    for k in k_values:
        # Get top-k for each ranking
        topk_sets = []
        for ranking in valid_rankings:
            topk = set(ranking[:min(k, len(ranking))])
            topk_sets.append(topk)
        
        # Compute pairwise Jaccard and average
        n = len(topk_sets)
        jaccards = []
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(topk_sets[i] & topk_sets[j])
                union = len(topk_sets[i] | topk_sets[j])
                if union > 0:
                    jaccards.append(intersection / union)
        
        results[k] = np.mean(jaccards) if jaccards else np.nan
    
    return results


def compute_top1_agreement(rankings_list):
    """
    Count how many sub-alignments agree on the top-1 mutation.
    
    Args:
        rankings_list: List of rankings
    
    Returns:
        Maximum agreement count (how many rankings share the same top-1)
    """
    valid_rankings = [r for r in rankings_list if r is not None and len(r) > 0]
    if len(valid_rankings) == 0:
        return 0
    
    # Get top-1 from each ranking (skip the initial D189S mutation at index 0)
    top1_mutations = []
    for ranking in valid_rankings:
        # Index 0 is the initial mutation (D189S), so index 1 is the first predicted
        if len(ranking) > 1:
            top1_mutations.append(ranking[1])
    
    if not top1_mutations:
        return 0
    
    # Count occurrences
    from collections import Counter
    counts = Counter(top1_mutations)
    
    return max(counts.values())

#######################################################

####################### MAIN GRID SEARCH #######################

def run_grid_search(skip_training=False):
    """
    Run the complete grid search.
    
    Args:
        skip_training: If True, only evaluate existing models without training
    
    Returns:
        List of result dictionaries
    """
    # Create results directory
    GRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    param_combinations = get_param_combinations()
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Sub-alignments to evaluate: {SUBALN_INDICES}")
    print(f"Total training runs: {len(param_combinations) * len(SUBALN_INDICES)}")
    print("=" * 60)
    
    results = []
    
    for combo_idx, params in enumerate(param_combinations):
        print(f"\n[{combo_idx + 1}/{len(param_combinations)}] Evaluating params: {params}")
        
        # Train/load models for all sub-alignments
        model_paths = []
        rankings = []
        
        for subaln_idx in SUBALN_INDICES:
            if not skip_training:
                model_path = train_single_model(subaln_idx, params, skip_if_exists=True)
            else:
                # Try to find existing model
                family_name = f"GridEval_SubAln_{subaln_idx}"
                model_dir = RESULTS_DIR / family_name
                expected_filename = get_model_filename(family_name, params)
                model_path = model_dir / expected_filename
                if not model_path.exists():
                    model_path = None
            
            model_paths.append(model_path)
            
            # Extract mutation ranking
            if model_path and model_path.exists():
                ranking = extract_mutation_ranking(model_path)
                rankings.append(ranking)
            else:
                rankings.append(None)
        
        # Compute robustness metrics
        valid_count = sum(1 for r in rankings if r is not None)
        
        if valid_count >= 2:
            mean_tau, tau_matrix = compute_kendall_tau_matrix(rankings)
            topk_jaccard = compute_topk_jaccard(rankings)
            top1_agreement = compute_top1_agreement(rankings)
        else:
            mean_tau = np.nan
            tau_matrix = None
            topk_jaccard = {3: np.nan, 5: np.nan, 10: np.nan}
            top1_agreement = 0
        
        result = {
            'params': params,
            'N_chains': params['N_chains'],
            'N_iter': params['N_iter'],
            'k_MCMC': params['k_MCMC'],
            'theta': params['theta'],
            'lambda_J': params['lambda_J'],
            'valid_models': valid_count,
            'mean_kendall_tau': mean_tau,
            'topk_jaccard_3': topk_jaccard.get(3, np.nan),
            'topk_jaccard_5': topk_jaccard.get(5, np.nan),
            'topk_jaccard_10': topk_jaccard.get(10, np.nan),
            'top1_agreement': top1_agreement,
            'rankings': rankings,
        }
        
        results.append(result)
        
        print(f"  Valid models: {valid_count}/{len(SUBALN_INDICES)}")
        print(f"  Mean Kendall's tau: {mean_tau:.3f}" if not np.isnan(mean_tau) else "  Mean Kendall's tau: N/A")
        print(f"  Top-5 Jaccard: {topk_jaccard.get(5, np.nan):.3f}" if not np.isnan(topk_jaccard.get(5, np.nan)) else "  Top-5 Jaccard: N/A")
        print(f"  Top-1 Agreement: {top1_agreement}/{len(SUBALN_INDICES)}")
    
    return results

#######################################################

####################### SUMMARY & VISUALIZATION #######################

def generate_summary(results):
    """
    Generate summary CSV and identify best parameter combinations.
    
    Args:
        results: List of result dictionaries from run_grid_search
    """
    # Save to CSV
    csv_path = GRID_RESULTS_DIR / 'grid_evaluation_summary.csv'
    
    fieldnames = [
        'N_chains', 'N_iter', 'k_MCMC', 'theta', 'lambda_J',
        'valid_models', 'mean_kendall_tau', 
        'topk_jaccard_3', 'topk_jaccard_5', 'topk_jaccard_10',
        'top1_agreement'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {k: r[k] for k in fieldnames}
            # Handle NaN values for CSV
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = ''
            writer.writerow(row)
    
    print(f"\nSummary saved to: {csv_path}")
    
    # Find best parameter combinations
    print("\n" + "=" * 60)
    print("TOP 5 MOST ROBUST PARAMETER COMBINATIONS")
    print("(Ranked by Mean Kendall's Tau)")
    print("=" * 60)
    
    # Filter out results with NaN tau
    valid_results = [r for r in results if not np.isnan(r['mean_kendall_tau'])]
    
    if not valid_results:
        print("No valid results to rank!")
        return
    
    # Sort by mean Kendall's tau (descending)
    sorted_results = sorted(valid_results, key=lambda x: x['mean_kendall_tau'], reverse=True)
    
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n#{i + 1}:")
        print(f"  Parameters:")
        print(f"    N_chains: {r['N_chains']}")
        print(f"    N_iter: {r['N_iter']}")
        print(f"    k_MCMC: {r['k_MCMC']}")
        print(f"    theta: {r['theta']}")
        print(f"    lambda_J: {r['lambda_J']}")
        print(f"  Metrics:")
        print(f"    Mean Kendall's tau: {r['mean_kendall_tau']:.4f}")
        print(f"    Top-3 Jaccard: {r['topk_jaccard_3']:.4f}")
        print(f"    Top-5 Jaccard: {r['topk_jaccard_5']:.4f}")
        print(f"    Top-10 Jaccard: {r['topk_jaccard_10']:.4f}")
        print(f"    Top-1 Agreement: {r['top1_agreement']}/{len(SUBALN_INDICES)}")
    
    # Save detailed results
    np.save(GRID_RESULTS_DIR / 'grid_evaluation_results.npy', results, allow_pickle=True)
    print(f"\nDetailed results saved to: {GRID_RESULTS_DIR / 'grid_evaluation_results.npy'}")
    
    return sorted_results


def generate_heatmap(results):
    """
    Generate a heatmap visualization of the grid search results.
    
    Args:
        results: List of result dictionaries
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("matplotlib not available, skipping heatmap generation")
        return
    
    # Create a pivot table for visualization
    # We'll create multiple heatmaps for different parameter pairs
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Grid Search Results: Mean Kendall\'s Tau', fontsize=14, fontweight='bold')
    
    # Extract unique values for each parameter
    n_chains_vals = sorted(set(r['N_chains'] for r in results))
    n_iter_vals = sorted(set(r['N_iter'] for r in results))
    k_mcmc_vals = sorted(set(r['k_MCMC'] for r in results))
    theta_vals = sorted(set(r['theta'] for r in results))
    lambda_j_vals = sorted(set(r['lambda_J'] for r in results))
    
    # Heatmap 1: N_chains vs N_iter (averaged over other params)
    ax1 = axes[0, 0]
    heatmap1 = np.zeros((len(n_chains_vals), len(n_iter_vals)))
    counts1 = np.zeros_like(heatmap1)
    
    for r in results:
        if not np.isnan(r['mean_kendall_tau']):
            i = n_chains_vals.index(r['N_chains'])
            j = n_iter_vals.index(r['N_iter'])
            heatmap1[i, j] += r['mean_kendall_tau']
            counts1[i, j] += 1
    
    heatmap1 = np.divide(heatmap1, counts1, where=counts1 > 0)
    heatmap1[counts1 == 0] = np.nan
    
    im1 = ax1.imshow(heatmap1, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(n_iter_vals)))
    ax1.set_xticklabels(n_iter_vals)
    ax1.set_yticks(range(len(n_chains_vals)))
    ax1.set_yticklabels(n_chains_vals)
    ax1.set_xlabel('N_iter')
    ax1.set_ylabel('N_chains')
    ax1.set_title('N_chains vs N_iter')
    plt.colorbar(im1, ax=ax1)
    
    # Heatmap 2: k_MCMC vs theta
    ax2 = axes[0, 1]
    heatmap2 = np.zeros((len(k_mcmc_vals), len(theta_vals)))
    counts2 = np.zeros_like(heatmap2)
    
    for r in results:
        if not np.isnan(r['mean_kendall_tau']):
            i = k_mcmc_vals.index(r['k_MCMC'])
            j = theta_vals.index(r['theta'])
            heatmap2[i, j] += r['mean_kendall_tau']
            counts2[i, j] += 1
    
    heatmap2 = np.divide(heatmap2, counts2, where=counts2 > 0)
    heatmap2[counts2 == 0] = np.nan
    
    im2 = ax2.imshow(heatmap2, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(theta_vals)))
    ax2.set_xticklabels(theta_vals)
    ax2.set_yticks(range(len(k_mcmc_vals)))
    ax2.set_yticklabels(k_mcmc_vals)
    ax2.set_xlabel('theta')
    ax2.set_ylabel('k_MCMC')
    ax2.set_title('k_MCMC vs theta')
    plt.colorbar(im2, ax=ax2)
    
    # Heatmap 3: N_chains vs lambda_J
    ax3 = axes[1, 0]
    heatmap3 = np.zeros((len(n_chains_vals), len(lambda_j_vals)))
    counts3 = np.zeros_like(heatmap3)
    
    for r in results:
        if not np.isnan(r['mean_kendall_tau']):
            i = n_chains_vals.index(r['N_chains'])
            j = lambda_j_vals.index(r['lambda_J'])
            heatmap3[i, j] += r['mean_kendall_tau']
            counts3[i, j] += 1
    
    heatmap3 = np.divide(heatmap3, counts3, where=counts3 > 0)
    heatmap3[counts3 == 0] = np.nan
    
    im3 = ax3.imshow(heatmap3, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(lambda_j_vals)))
    ax3.set_xticklabels(lambda_j_vals)
    ax3.set_yticks(range(len(n_chains_vals)))
    ax3.set_yticklabels(n_chains_vals)
    ax3.set_xlabel('lambda_J')
    ax3.set_ylabel('N_chains')
    ax3.set_title('N_chains vs lambda_J')
    plt.colorbar(im3, ax=ax3)
    
    # Heatmap 4: Top-5 Jaccard by N_chains vs k_MCMC
    ax4 = axes[1, 1]
    heatmap4 = np.zeros((len(n_chains_vals), len(k_mcmc_vals)))
    counts4 = np.zeros_like(heatmap4)
    
    for r in results:
        if not np.isnan(r['topk_jaccard_5']):
            i = n_chains_vals.index(r['N_chains'])
            j = k_mcmc_vals.index(r['k_MCMC'])
            heatmap4[i, j] += r['topk_jaccard_5']
            counts4[i, j] += 1
    
    heatmap4 = np.divide(heatmap4, counts4, where=counts4 > 0)
    heatmap4[counts4 == 0] = np.nan
    
    im4 = ax4.imshow(heatmap4, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(k_mcmc_vals)))
    ax4.set_xticklabels(k_mcmc_vals)
    ax4.set_yticks(range(len(n_chains_vals)))
    ax4.set_yticklabels(n_chains_vals)
    ax4.set_xlabel('k_MCMC')
    ax4.set_ylabel('N_chains')
    ax4.set_title('Top-5 Jaccard: N_chains vs k_MCMC')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    
    heatmap_path = GRID_RESULTS_DIR / 'grid_evaluation_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {heatmap_path}")

#######################################################

####################### MAIN #######################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid evaluation for SBM hyperparameters')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only evaluate existing models')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on previously saved results')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Load previous results
        results_path = GRID_RESULTS_DIR / 'grid_evaluation_results.npy'
        if results_path.exists():
            print(f"Loading results from {results_path}")
            results = np.load(results_path, allow_pickle=True).tolist()
            generate_summary(results)
            generate_heatmap(results)
        else:
            print(f"No previous results found at {results_path}")
    else:
        # Run full grid search
        print("Starting Grid Evaluation for SBM Hyperparameters")
        print("=" * 60)
        
        results = run_grid_search(skip_training=args.skip_training)
        
        # Generate summary and visualizations
        generate_summary(results)
        generate_heatmap(results)
        
        print("\n" + "=" * 60)
        print("Grid evaluation complete!")
        print("=" * 60)
