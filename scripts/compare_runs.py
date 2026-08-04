import json
from pathlib import Path

def main():
    exp_dir = Path(__file__).parent.parent / "experiments"
    
    if not exp_dir.exists():
        print("No experiments directory found.")
        return
        
    print("| Model | AUROC  | AUPRC  | Utility | F1 | Train Time |")
    print("| ----- | ------ | ------ | ------- | -- | ---------- |")
    
    for model_dir in exp_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Find all runs for this model
        best_run = None
        best_auroc = -1.0
        
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
                
            with open(metrics_path, "r") as f:
                try:
                    metrics = json.load(f)
                    auroc = metrics.get("auroc", 0)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_run = metrics
                except Exception:
                    pass
                    
        if best_run:
            auroc = f"{best_run.get('auroc', 0):.4f}"
            auprc = f"{best_run.get('auprc', 0):.4f}"
            utility = f"{best_run.get('utility_score', 0):.2f}"
            f1 = f"{best_run.get('f1', 0):.4f}"
            # Time isn't saved in metrics currently, maybe grab from history.csv later
            time_str = "—" 
            print(f"| {model_name} | {auroc} | {auprc} | {utility} | {f1} | {time_str} |")

if __name__ == "__main__":
    main()
