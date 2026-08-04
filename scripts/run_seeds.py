import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tact.yaml")
    args = parser.parse_args()
    
    seeds = [42, 123, 256, 777, 999, 2026, 4096]
    
    # Read the base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
        
    print(f"Using configuration from: {args.config}")
    
    all_results = []
    
    for seed in seeds:
        print(f"\n======================================")
        print(f"  RUNNING M3 | SEED: {seed}")
        print(f"======================================")
        
        # We need to run train.py
        # It creates a run directory in experiments/tact/run_YYYYMMDD_HHMMSS
        # We will parse the output directory to find metrics.json
        
        cmd = [
            "python", "scripts/train.py",
            "--config", args.config,
            "--seed", str(seed)
        ]
        
        try:
            # Capture the output to extract the experiment directory
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout
            print(output)
            
            # Find the metrics.json in the latest experiments/tact/ directory
            import glob
            exp_dirs = glob.glob("experiments/tact/run_*")
            if not exp_dirs:
                print(f"ERROR: Could not find output directory for seed {seed}")
                continue
                
            latest_exp_dir = max(exp_dirs, key=os.path.getctime)
            metrics_path = os.path.join(latest_exp_dir, "metrics.json")
            
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    all_results.append({
                        "seed": seed,
                        "utility": metrics["utility_score"],
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "f1": metrics["f1"]
                    })
            else:
                print(f"ERROR: metrics.json not found in {latest_exp_dir}")
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR running seed {seed}:")
            print(e.stderr)
            print(e.stdout)
            
    print("\n======================================")
    print("  MULTI-SEED EVALUATION COMPLETE      ")
    print("======================================")
    
    if not all_results:
        print("No valid results collected.")
        return
        
    # Calculate Mean and Std
    utilities = [r["utility"] for r in all_results]
    aurocs = [r["auroc"] for r in all_results]
    auprcs = [r["auprc"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    
    print(f"Total Runs: {len(all_results)} / {len(seeds)}")
    print(f"Utility : {np.mean(utilities):.4f} ± {np.std(utilities):.4f}")
    print(f"AUROC   : {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"AUPRC   : {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"F1      : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # Save the aggregated results
    with open("multi_seed_results.json", "w") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "utility_mean": np.mean(utilities),
                "utility_std": np.std(utilities),
                "auroc_mean": np.mean(aurocs),
                "auroc_std": np.std(aurocs),
                "auprc_mean": np.mean(auprcs),
                "auprc_std": np.std(auprcs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            }
        }, f, indent=4)
        
if __name__ == "__main__":
    main()
