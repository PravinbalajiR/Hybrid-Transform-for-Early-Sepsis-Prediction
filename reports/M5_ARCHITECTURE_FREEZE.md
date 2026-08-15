# Phase 1: M5 Architecture Freeze & Specification Report

**Model Name:** M5 — Multi-Hybrid Time-Aware Sepsis Intelligence Network  
**Freeze Status:** **FROZEN (ZERO ARCHITECTURAL REDESIGN)**  
**Total Parameters:** `224,713`  
**Trainable Parameters:** `224,713`  

---

## 1. Executive Summary

This document freezes the exact mathematical and neural architecture of **M5**. In accordance with the scientific research protocol, M5's structural modules, parameter projections, and tensor flows are locked without modification.

---

## 2. Module Specifications

```
========================================================================================
             M5: MULTI-HYBRID TIME-AWARE SEPSIS INTELLIGENCE NETWORK
========================================================================================

                 Input Triplet Vector [Batch × SeqLen × 102]
                  │ 34 Values, 34 Masks, 34 Time Deltas
                  ▼
   ┌──────────────────────┬──────────────────────┬──────────────────────┐
   │ Value Encoder        │ Mask Encoder         │ Time Encoder (log1p) │
   │ (Linear 34 → 32)     │ (Linear 34 → 32)     │ (Linear 34 → 32)     │
   └──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
              │                      │                      │
              ▼                      ▼                      ▼
        h_val [dim=64]         h_mask [dim=64]        h_time [dim=64]
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                                     ▼
                            Shared Representation
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
     Local Expert             Global Expert            Time-Aware Expert
   (Causal Conv1D TCN)    (Causal Transformer)     (Irregular Sampling MLP)
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     ▼
                      MoE Adaptive Gating Router
                        w = Softmax(G(h_shared))
                        w1 + w2 + w3 = 1.0
                                     │
                                     ▼
                       Adaptive Representation Fusion
                      Softmax([h_val, h_mask, h_time, h_exp])
                                     │
                                     ▼
                       Temporal Attention Pooling
                        Causal Attention Scores
                                     │
                                     ▼
                     Classifier Head (MLP → Sigmoid)
                                     │
                                     ▼
                          Hourly Sepsis Probability
```

---

## 3. Parameter Breakdown by Module

| Module Name | Component Class | Parameter Count | Percentage of Total |
|---|---|:---:|:---:|
| **Value Encoder** | `ValueEncoder` + `val_proj` | $34 \times 32 + 32 + 32 \times 64 + 64 = 3,296$ | 1.47% |
| **Mask Encoder** | `MaskEncoder` + `mask_proj` | $34 \times 32 + 32 + 32 \times 64 + 64 = 3,296$ | 1.47% |
| **Time Encoder** | `TimeEncoder` + `time_proj` | $34 \times 32 + 32 + 32 \times 64 + 64 = 3,296$ | 1.47% |
| **Shared Projection** | `shared_proj` | $96 \times 64 + 64 = 6,208$ | 2.76% |
| **Local Temporal Expert** | `LocalTemporalExpert` (Causal TCN) | $64 \times 64 \times 3 + 64 \times 64 \times 3 + \dots = 37,248$ | 16.57% |
| **Global Temporal Expert** | `GlobalTemporalExpert` (Transformer) | 3 Layers $\times$ Transformer Layer ($d=64, h=4$) $= 99,840$ | 44.42% |
| **Time-Aware Expert** | `TimeAwareExpert` (MLP) | $(32+64) \times 64 + 64 \times 64 = 10,432$ | 4.64% |
| **MoE Router** | `MoERouter` | $64 \times 32 + 32 \times 3 = 2,179$ | 0.97% |
| **Adaptive Fusion** | `AdaptiveRepresentationFusion` | $256 \times 32 + 32 \times 4 + 64 \times 64 = 12,516$ | 5.57% |
| **Temporal Attention Pooling** | `TemporalAttentionPooling` | $64 \times 32 + 32 \times 1 = 2,081$ | 0.93% |
| **Prediction Head** | `classifier` | $64 \times 32 + 32 \times 1 + 33 = 2,113$ | 0.94% |
| **Total** | `M5Model` | **224,713** | **100.0%** |

---

## 4. Forward-Pass Tensor Shape Contract

Given an hourly ICU batch tensor of size `[Batch, SeqLen, 102]`:
1. `x_val` `[B, T, 34]` $\to$ `h_val` `[B, T, 64]`
2. `x_mask` `[B, T, 34]` $\to$ `h_mask` `[B, T, 64]`
3. `x_delta` `[B, T, 34]` $\to$ `log1p(clamp(x_delta, 0, 168))` $\to$ `h_time` `[B, T, 64]`
4. `h_shared` = `Linear(cat([e_val, e_mask, e_time]))` `[B, T, 64]`
5. `e_local` = `CausalConv1d(h_shared)` `[B, T, 64]`
6. `e_global` = `CausalTransformer(h_shared, causal_mask)` `[B, T, 64]`
7. `e_time` = `TimeAwareMLP(e_time, h_val + h_mask)` `[B, T, 64]`
8. `w` = `Softmax(Gating(h_shared))` `[B, T, 3]`; `h_expert` = $w_1 e_{local} + w_2 e_{global} + w_3 e_{time}$ `[B, T, 64]`
9. `h_fused` = `AdaptiveFusion([h_val, h_mask, h_time, h_expert])` `[B, T, 64]` (Softmax branch weights $\sum w = 1.0$)
10. `h_final` = `CausalAttentionPooling(h_fused)` `[B, T, 64]`
11. `logits` = `Classifier(h_final)` `[B, T]`; `probs` = `Sigmoid(logits)` `[B, T]`
