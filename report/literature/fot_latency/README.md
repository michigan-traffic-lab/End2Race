# FOT latency literature

This folder contains the directly relevant embedded-hardware benchmarks for
the Frenet Optimal Trajectory (FOT) planner family used by the End2Race
expert. The papers measure the planner; they do not include Pure Pursuit in
the reported planner latency.

## Primary source for the approximately 10 ms figure

Filippo Muzzini, Nicola Capodieci, Federico Ramanzin, and Paolo Burgio,
"GPU Implementation of the Frenet Path Planner for Embedded Autonomous
Systems: A Case Study in the F1TENTH Scenario," *Journal of Systems
Architecture*, vol. 154, article 103239, 2024.

- DOI: https://doi.org/10.1016/j.sysarc.2024.103239
- Open-access source: https://air.unipr.it/handle/11381/3014813
- Local file: `muzzini_et_al_2024_gpu_frenet_path_planner_f1tenth.pdf`
- F1TENTH workload: 240 candidate paths with 21 points per path
- Hardware: NVIDIA Jetson Xavier NX
- CPU planner-node rate: 98--110 Hz, equivalent to approximately
  10.20--9.09 ms per published trajectory
- GPU planner-node rate: 170--200 Hz, equivalent to approximately
  5.88--5.00 ms per published trajectory
- Scope: the complete ROS planner node, including message reception and
  publication overhead, but excluding downstream path tracking/control

The same paper also reports 68.48 ms for its C++ CPU baseline and 7.75 ms
for its full GPU implementation using a much denser benchmark of 1,024 paths
with 64 points each on an NVIDIA Xavier AGX.

## Preliminary paper

Filippo Muzzini, Nicola Capodieci, Federico Ramanzin, and Paolo Burgio,
"Optimized Local Path Planner Implementation for GPU-Accelerated Embedded
Systems," *IEEE Embedded Systems Letters*, 2023.

- DOI: https://doi.org/10.1109/LES.2023.3298733
- Local file: `muzzini_et_al_2023_optimized_local_path_planner_gpu.pdf`

This is the shorter preliminary version of the work. Use the extended 2024
journal paper for the F1TENTH-specific planner-node frequencies and the
approximately 10 ms literature claim.

## Safe wording

> Prior benchmarking on a Jetson Xavier NX reports approximately 9--10 ms
> per CPU-based FOT planner-node update for 240 candidate trajectories with
> 21 points each, including ROS communication overhead.

This result should not be described as a measurement of the End2Race Python
expert or as end-to-end sensing-to-actuation latency.
