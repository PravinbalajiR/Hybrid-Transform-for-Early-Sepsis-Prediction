# 📐 PHYSIONET 2019 OFFICIAL UTILITY SCORE FORMULATION

## Metric Definition
The official PhysioNet 2019 Utility metric evaluates clinical alarm timing for sepsis early warning.

- **Early Warning Window:** $[t_{onset} - 12	ext{h}, t_{onset} - 6	ext{h}]$ (linear credit from 0.0 to 1.0)
- **Optimal Warning Window:** $[t_{onset} - 6	ext{h}, t_{onset} + 3	ext{h}]$ (decay credit from 1.0 to 0.0)
- **False Alarm Penalty:** $-0.05$ points per hour for alarms issued before $t_{onset} - 12	ext{h}$ or for non-septic patients.
- **Missed Sepsis Penalty:** $-2.0$ points if no alarm is issued for a septic patient.
- **Cohort Normalization:** $	ext{Utility} = rac{\sum 	ext{Achieved Utility}}{\sum 	ext{Best Possible Utility}}$, where Best Possible Utility = $N_{	ext{sepsis}} 	imes 1.0$.
