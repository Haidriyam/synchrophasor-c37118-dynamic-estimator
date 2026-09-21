![Synchrophasor Dynamic Estimator CI](https://github.com/Haidriyam/synchrophasor-c37118-dynamic-estimator/actions/workflows/devsecops-ci.yml/badge.svg)

# IEEE C37.118 Synchrophasor Telemetry Dissector & Dynamic Generator EKF

A real-time synchrophasor telemetry ingest and dynamic state estimation pipeline. It couples a low-overhead binary frame parser for IEEE C37.118.2-2011 streaming telemetry with a discrete 4th-order Extended Kalman Filter (EKF) that tracks synchronous generator internal transient states ($\delta, \Delta\omega, E'_q, E'_d$) from voltage and current phasors.

```text
[ PMU Stream / UDP ] ──► [ IEEE C37.118 Frame Dissector ] ──► [ Integrity Filter ]
                                                                       │
                                                                       ▼
[ Rotor States (δ, ω) ] ◄── [ Non-Linear EKF Estimator ] ◄── [ Terminal (Vt, Pe, Qe) ]