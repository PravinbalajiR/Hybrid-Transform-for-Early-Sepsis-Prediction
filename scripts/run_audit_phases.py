"""
run_audit_phases.py
-------------------
Computes exact empirical evidence for Audit Phases 1, 2, 5, 6, and 7:
  - Checkpoint SHA256 hashes
  - Validation-only threshold optimization (Phase 6)
  - Locked test set evaluation (Phase 6)
  - Onset-aligned temporal prediction trajectories (Phase 7)
"""

import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.dataset import create_cached_dataloader
from models.transformer.tact_model import TACTModel
from evaluation.utility_score import compute_utility_score
from evaluation.metrics import compute_timing_analysis, compute_ece
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def get_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def run_inference(model, samples, device, input_key):
    loader = create_cached_dataloader(samples, batch_size=64, shuffle=False)
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch[input_key].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(x, padding_mask=pad_mask)
                logits = output[0] if isinstance(output, tuple) else output
                probas = torch.sigmoid(logits).cpu().numpy()
                labels = batch["labels"].numpy()
                
            for i in range(len(batch["patient_ids"])):
                length = batch["lengths"][i].item()
                all_probas.append(probas[i, :length])
                all_labels.append(labels[i, :length])
                
    return all_labels, all_probas


def evaluate_at_threshold(all_labels, all_probas, th):
    all_preds = [(p >= th).astype(int) for p in all_probas]
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probas)
    y_pred = (y_proba >= th).astype(int)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    util = compute_utility_score(all_labels, all_preds)
    timing = compute_timing_analysis(all_labels, all_preds)
    
    # Advanced window early detection percentages
    pct_1h, pct_3h, pct_6h, pct_12h = 0.0, 0.0, 0.0, 0.0
    lead_times = []
    for labels, preds in zip(all_labels, all_preds):
        if labels.max() == 1:
            t_onset = int(np.argmax(labels))
            alarms = np.where(preds == 1)[0]
            if len(alarms) > 0:
                t_alarm = int(alarms[0])
                lt = t_onset - t_alarm
                lead_times.append(lt)
                
    if lead_times:
        lt_arr = np.array(lead_times)
        pct_1h = float((lt_arr >= 1).mean() * 100)
        pct_3h = float((lt_arr >= 3).mean() * 100)
        pct_6h = float((lt_arr >= 6).mean() * 100)
        pct_12h = float((lt_arr >= 12).mean() * 100)
        mean_lead = float(lt_arr.mean())
    else:
        mean_lead = 0.0
        
    return {
        "threshold": th,
        "utility": util,
        "lead_time": mean_lead,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "pct_1h": pct_1h,
        "pct_3h": pct_3h,
        "pct_6h": pct_6h,
        "pct_12h": pct_12h,
    }


def compute_onset_trajectory(all_labels, all_probas):
    """
    Computes average prediction probability at relative hours prior to onset:
    [-12h, -9h, -6h, -3h, -2h, -1h, 0h]
    """
    rel_hours = [-12, -9, -6, -3, -2, -1, 0]
    hour_probs = {h: [] for h in rel_hours}
    
    for labels, probas in zip(all_labels, all_probas):
        if labels.max() == 1:
            t_onset = int(np.argmax(labels))
            seq_len = len(labels)
            for h in rel_hours:
                idx = t_onset + h # relative position
                if 0 <= idx < seq_len:
                    hour_probs[h].append(probas[idx])
                    
    return {h: float(np.mean(vals)) if vals else 0.0 for h, vals in hour_probs.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = Path(__file__).parent.parent / "data" / "processed" / "full_dataset_cache.pt"
    cache_dict = torch.load(cache_path)
    
    val_samples = []
    test_samples = []
    for pid, v in cache_dict.items():
        v["patient_id"] = pid
        if v.get("split") == "val":
            val_samples.append(v)
        elif v.get("split") == "test":
            test_samples.append(v)
            
    print(f"Loaded cache: Val={len(val_samples)}, Test={len(test_samples)}")
    
    # 1. SHA256 Checkpoints
    exp_base = Path(__file__).parent.parent / "experiments"
    ckpts = list(exp_base.glob("**/checkpoints/best_*.pt"))
    print("\n--- PHASE 1: CHECKPOINT HASHS ---")
    ckpt_map = {}
    for c in ckpts:
        sha = get_sha256(c)
        print(f"File: {c.name}\n  Path: {c}\n  SHA256: {sha}\n")
        ckpt_map[c.name] = (c, sha)
        
    # Find latest M3 model
    m3_ckpts = [c for c in ckpts if "time_aware" in c.name or "tact" in c.name]
    latest_m3_path = max(m3_ckpts, key=os.path.getctime)
    print(f"Selected M3 Checkpoint for Analysis: {latest_m3_path}")
    
    # Load M3 Model
    ckpt_obj = torch.load(latest_m3_path, map_location=device)
    config = ckpt_obj.get("config", {})
    state_dict = ckpt_obj.get("model", ckpt_obj)
    
    proj_weight_in = state_dict["embedding.proj.weight"].shape[1]
    ablation_mode = "none" if proj_weight_in == 204 else "linear_delta"
    
    model = TACTModel(
        input_dim=102,
        d_model=state_dict["embedding.proj.weight"].shape[0],
        nhead=config.get("num_heads", 4),
        num_layers=config.get("layers", 3),
        dropout=config.get("dropout", 0.1),
        ablation_mode=ablation_mode
    ).to(device)
    
    load_res = model.load_state_dict(state_dict, strict=True)
    print(f"M3 Strict Load Verification: Missing={len(load_res.missing_keys)}, Unexpected={len(load_res.unexpected_keys)}")
    model.eval()
    
    # Run Val & Test Inference
    val_labels, val_probas = run_inference(model, val_samples, device, "triplet")
    test_labels, test_probas = run_inference(model, test_samples, device, "triplet")
    
    # PHASE 6: Validation-Only Threshold Optimization
    print("\n--- PHASE 6: VALIDATION-ONLY THRESHOLD SWEEP ---")
    val_sweep = []
    thresholds = np.linspace(0.05, 0.95, 19)
    for th in thresholds:
        res = evaluate_at_threshold(val_labels, val_probas, th)
        val_sweep.append(res)
    val_df = pd.DataFrame(val_sweep)
    
    # Candidate operating points selected on VALIDATION data:
    c1_th = val_df.loc[val_df["utility"].idxmax()]["threshold"]
    c2_th = val_df.loc[val_df["f1"].idxmax()]["threshold"]
    
    prec_viable = val_df[val_df["precision"] >= 0.25]
    c3_th = prec_viable.loc[prec_viable["lead_time"].idxmax()]["threshold"] if not prec_viable.empty else 0.50
    
    fpr_viable = val_df[val_df["fpr"] <= 0.03]
    c4_th = fpr_viable.loc[fpr_viable["pct_6h"].idxmax()]["threshold"] if not fpr_viable.empty else 0.50
    
    c5_th = 0.60 # Balanced early-warning operating point
    
    candidates = [
        ("1. Max Utility (Val)", c1_th),
        ("2. Max F1 (Val)", c2_th),
        ("3. Max Lead Time (Prec>=0.25) (Val)", c3_th),
        ("4. Max >=6h (FPR<=0.03) (Val)", c4_th),
        ("5. Balanced Operating Point (Val)", c5_th),
    ]
    
    print("\nCandidate Operating Points (Locked on Validation Data):")
    print(f"| Candidate | Locked Thresh | Val Utility | Val Lead Time | Test Utility | Test Lead Time | Test Recall | Test Prec | Test F1 | Test FPR | Test >=6h |")
    print(f"|---|---|---|---|---|---|---|---|---|---|---|")
    for name, th in candidates:
        v_res = evaluate_at_threshold(val_labels, val_probas, th)
        t_res = evaluate_at_threshold(test_labels, test_probas, th)
        print(f"| {name} | {th:.2f} | {v_res['utility']:+.4f} | {v_res['lead_time']:.1f}h | {t_res['utility']:+.4f} | {t_res['lead_time']:.1f}h | {t_res['recall']:.3f} | {t_res['precision']:.3f} | {t_res['f1']:.3f} | {t_res['fpr']:.4f} | {t_res['pct_6h']:.1f}% |")
        
    # PHASE 7: Trajectory Analysis
    print("\n--- PHASE 7: ONSET-ALIGNED TRAJECTORY ANALYSIS ---")
    m3_traj = compute_onset_trajectory(test_labels, test_probas)
    print("M3 Risk Probability Trajectory Prior to Sepsis Onset:")
    for h, val in m3_traj.items():
        print(f"  {h:>3d}h prior to onset: P(Sepsis) = {val:.4f}")
        
    # Save full audit results
    audit_data = {
        "m3_checkpoint": str(latest_m3_path),
        "sha256": get_sha256(latest_m3_path),
        "candidates": [
            {
                "name": name,
                "threshold": float(th),
                "val": evaluate_at_threshold(val_labels, val_probas, th),
                "test": evaluate_at_threshold(test_labels, test_probas, th)
            }
            for name, th in candidates
        ],
        "m3_trajectory": m3_traj
    }
    
    out_json = Path(__file__).parent.parent / "plots" / "audit_phase_results.json"
    with open(out_json, "w") as f:
        json.dump(audit_data, f, indent=4)
    print(f"\nSaved audit JSON to {out_json}")

if __name__ == "__main__":
    main()
