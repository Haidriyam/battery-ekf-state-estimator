![BMS State Estimator CI](https://github.com/Haidriyam/bms-ekf-state-estimator/actions/workflows/devsecops-ci.yml/badge.svg)

# Dynamic SoC Extended Kalman Filter (EKF) with CAN-Bus Ingestion Fuzzing

A real-time state estimation and input fuzzing suite for Electric Vehicle (EV) Battery Management Systems (BMS). Combines discrete nonlinear Kalman filtering for internal electrochemical state estimation with robust, boundary-enforced CAN-bus telemetry parsing.

```text
[ Raw CAN Frame: 0x3B4 ] ──► [ Boundary Fuzzer / Parser ] ──► (V_term, I_pack)
                                                                    │
                                                                    ▼
[ Thevenin 1-RC Plant ] ◄── [ Extended Kalman Filter ] ◄── (State Predict & Update)
                                      │
                         [ Estimated SoC & Polarized Overpotential ]