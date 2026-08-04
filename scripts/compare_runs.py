import json
from pathlib import Path

def main():
    exp_dir = Path(__file__).parent.parent / "experiments"
    
    if not exp_dir.exists():
        print("No experiments directory found.")
        return
        
    print("\n=========================================================================================================")
    print("                                MASTER EXPERIMENTAL COMPARISON TABLE                                      ")
    print("=========================================================================================================")
    print("| Model | Utility | AUROC | AUPRC | F1 | Precision | Recall | ECE | Mean Lead Time |")
    print("| ----- | ------- | ----- | ----- | -- | --------- | ------ | --- | -------------- |")
    
    for model_dir in exp_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Find latest run for this model
        best_run = None
        latest_time = 0
        
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
                
            mtime = metrics_path.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                with open(metrics_path, "r") as f:
                    try:
                        best_run = json.load(f)
                    except Exception:
                        pass
                    
        if best_run:
            utility = f"{best_run.get('utility_score', 0):.4f}"
            auroc   = f"{best_run.get('auroc', 0):.4f}"
            auprc   = f"{best_run.get('auprc', 0):.4f}"
            f1      = f"{best_run.get('f1', 0):.4f}"
            prec    = f"{best_run.get('precision', 0):.4f}"
            rec     = f"{best_run.get('recall', 0):.4f}"
            ece     = f"{best_run.get('ece', 0):.4f}"
            
            lead_h  = best_run.get('timing_mean_lead_h', None)
            lead_str = f"{lead_h:.1f}h" if lead_h is not None else "N/A"
            
            print(f"| {model_name:12s} | {utility:7s} | {auroc:5s} | {auprc:5s} | {f1:4s} | {prec:9s} | {rec:6s} | {ece:3s} | {lead_str:14s} |")

if __name__ == "__main__":
    main()

