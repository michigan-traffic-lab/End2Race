# F1TENTH / RoboRacer learning literature review

Search completed **2026-08-13**. This folder contains **53 verified PDFs** on F1TENTH/RoboRacer imitation learning (IL), reinforcement learning (RL), single-vehicle racing, head-to-head racing, and closely related methods. Seven additional papers are documented below because their publishers blocked automated download.

## Scope and reading guide

The core set requires an experiment in F1TENTH/RoboRacer hardware, its simulator, or an explicitly F1TENTH-style 1:10 platform. I also retained a labeled **adjacent** set when it supplies a useful method or baseline: scaled autonomous racing on another platform, learned control without IL/RL, game-theoretic overtaking, or a safety layer evaluated on racing dynamics.

Abbreviations: **BC** = behavioral cloning; **DAgger** = Dataset Aggregation; **DR** = domain randomization; **MBRL** = model-based RL; **MFRL** = model-free RL; **PP** = pure pursuit; **CBF** = control barrier function; **H2H** = head-to-head.

The search began from the [F1TENTH RL benchmark](https://proceedings.mlr.press/v123/o-kelly20a.html), the [official F1TENTH research index](https://f1tenth.github.io/research.html), the 2024 F1TENTH survey, and the 2025 RoboRacer survey. I then followed citations and searched titles/abstracts for `F1TENTH`, `F1/10`, `RoboRacer`, imitation, reinforcement, residual policy, safe learning, overtaking, and head-to-head. Preprints and publisher versions of the same work were counted once where identifiable.

## Main findings

1. **The literature is still weighted toward single-car time trials.** Work through roughly 2022 primarily studied reward design, architecture selection, cross-track transfer, and sim-to-real deployment. H2H learning becomes a sustained topic from 2023 onward, but its evidence base is smaller.

2. **Hybrid and residual policies are the dominant practical deployment pattern.** TC-Driver, residual policy learning, RLPP, onboard residual SAC, and attenuated residual policy optimization preserve a classical trajectory follower or safety layer while learning the missing performance correction. This improves sample efficiency and deployment confidence, but can cap performance or inherit errors from the base policy. The 2026 attenuated-residual work explicitly tries to remove that dependency over training.

3. **IL is small but increasingly important.** The 2022 zero-shot comparison found a familiar trade-off: its RL policies were generally faster, while IL was safer. The 2023 benchmark found interactive IL more robust than plain BC and useful for initializing RL. MEGA-DAgger addresses inconsistent multiple experts for H2H overtaking, while TinyLidarNet and the FPGA neural controller show that expert imitation can produce very small, fast physical controllers. End2Race adds temporal modeling and interaction-aware LiDAR features.

4. **H2H results are promising but not yet standardized.** Bhargav et al. learn an offline overtaking prior and retain model-predictive control. Sense-Imagine-Act/Lucid Dreamer and later context-aware MBRL learn world models. MEGA-DAgger learns from experts. RaceMOP learns a mapless residual collision-avoidance policy. Steiner et al. and Cihlar et al. add valuable physical overtaking tests. These are not directly rankable: opponent policies, starting gaps, maps, vehicle speeds, collision rules, and episode definitions differ.

5. **Real-world evidence remains the bottleneck.** Many strong lap-time and overtaking numbers are simulation-only. The most useful deployment papers explicitly model actuator dynamics, latency, sensor noise, supervisory safety, or onboard adaptation. Physical H2H racing against diverse reactive opponents is still rare.

6. **Generalization is the unresolved scientific problem.** Cross-track zero-shot transfer is now common in single-car studies, but opponent-behavior shift is less mature. Context-aware MBRL directly targets unseen opponent behavior; the 2026 overtaking/pose-estimation paper reports that a policy successful under familiar conditions degrades on different track geometry. End2Race is strong on broad simulated scenario coverage but still needs hardware validation.

## Imitation learning and mixed IL/RL

| Year | Paper | Method and evidence | Relevance |
|---:|---|---|---|
| 2022 | [A comparison of reinforcement learning and imitation learning for zero-shot autonomous racing](hamilton_et_al_2022_zero_shot_rl_vs_il.pdf) | Compares IL and RL under zero-shot track transfer in F1TENTH-style racing. | Direct comparison; highlights performance/safety trade-off. |
| 2022 | [Evaluating the robustness of reinforcement learning and imitation learning for autonomous vehicles](schuman_et_al_2022_evolutionary_vs_il.pdf) | Evolutionary/RL and imitation-style learned driving comparison under perturbations. | Adjacent robustness evidence. |
| 2023 | [An imitation learning benchmark for autonomous racing](sun_et_al_2023_il_benchmark.pdf) | BC and interactive IL, with IL also used to bootstrap RL; F1TENTH simulation/hardware-oriented evaluation. | Best entry point for choosing an IL algorithm. |
| 2024 | [MEGA-DAgger: Imitation learning with multiple imperfect experts](sun_et_al_2024_mega_dagger.pdf) | Selects/filter-demonstrations from multiple experts; simulated and physical F1TENTH overtaking. | Core H2H IL paper; directly addresses expert inconsistency. |
| 2024 | [TinyLidarNet](zarrar_et_al_2024_tinylidarnet.pdf) | Lightweight 1-D LiDAR network trained from expert labels and deployed on constrained hardware. | Strong embedded-IL reference; physical competition validation. |
| 2024/25 | [Hardware neural control of an F1TENTH race car on FPGA](paluch_et_al_2024_fpga_neural_control.pdf) | Supervised imitation of nonlinear MPC, deployed at very high control rates on physical F1TENTH hardware. | Shows the latency/throughput advantage of compact imitation. |
| 2025 | [End2Race](end2race_2025.pdf) | GRU policy with interaction-aware spatial-pressure LiDAR tokens; single-car and H2H simulated scenarios. | Closest temporal end-to-end H2H reference for this repository. |
| 2026 | [Adaptive control in autonomous driving via real-time recurrent RL](lemmel_et_al_2026_realtime_recurrent_rl.pdf) | Offline BC followed by online real-time recurrent RL fine-tuning; event-camera 1:10 RoboRacer experiments. | Important bridge between imitation initialization and online adaptation. |

The clearest progression is **BC -> interactive/multi-expert data collection -> temporal policies -> online adaptation**. For a new H2H system, comparisons against plain BC alone would therefore be weak. A credible IL study should include at least BC, an interactive method such as DAgger, a multi-expert treatment if several planners provide labels, and an ablation on recurrent state.

## Single-vehicle reinforcement learning and learned control

| Year | Paper | Main idea | Evidence / caveat |
|---:|---|---|---|
| 2021 | [Learning local planning for autonomous racing using deep RL](evans_et_al_2021_learning_local_planning.pdf) | RL local planner for obstacle avoidance around a racing controller. | Direct F1TENTH line of work; hybrid rather than fully end-to-end. |
| 2021 | [Reward signal design for autonomous racing](evans_et_al_2021_reward_signal_design.pdf) | Controlled comparison of reward designs. | Trajectory-relative shaping is the key reusable lesson. |
| 2021 | [Hierarchical reward shaping for autonomous racing](berducci_et_al_2021_hierarchical_reward_shaping.pdf) | Hierarchical/shaped rewards for faster and safer learning. | Useful reward-design companion. |
| 2021/22 | [Model-based RL in latent space for autonomous racing from pixels](brunnbauer_et_al_2022_latent_imagination.pdf) | Dreamer-style latent imagination and zero-shot sim-to-real. | Foundational F1TENTH MBRL paper. |
| 2021 | [Feedback-linearization-aided reinforcement learning](estrada_et_al_2021_feedback_linearization_rl.pdf) | Combines control structure with learned policy improvement. | Adjacent hybrid-control reference. |
| 2022/23 | [TC-Driver: Trajectory-conditioned reinforcement learning](ghignone_et_al_2023_tc_driver.pdf) | Conditions the learned controller on a reference trajectory. | Strong hybrid sim-to-real/cross-track reference. |
| 2022/23 | [Residual policy learning for autonomous racing](trumpp_et_al_2023_residual_policy_learning.pdf) | Learns corrections to a conventional racing policy. | Multi-track simulation; reports mean lap-time improvement over its base policy. |
| 2023 | [Comparing deep RL architectures for autonomous racing](evans_et_al_2023_drl_architectures.pdf) | Compares end-to-end, trajectory-aided, and planning architectures with DDPG/TD3/SAC. | Full planning is strongest in its simulator; end-to-end transfers better in its real tests. |
| 2023 | [Trajectory-aided deep reinforcement learning](evans_et_al_2023_trajectory_aided_drl.pdf) | Supplies racing-line/trajectory information to the policy. | Useful intermediate point between end-to-end and modular planning. |
| 2023 | [Safe reinforcement learning for high-speed autonomous racing](evans_et_al_2023_safe_rl.pdf) | Viability-based supervisor prevents unsafe exploratory actions. | Learns rapidly without training crashes in the reported tests, at a speed cost. |
| 2023 | [Autonomous racing with multiple vehicles using a learning-based model predictive controller](kochdumper_et_al_2023_provably_safe_rl.pdf) | Learning-enhanced predictive control with formal/safety structure. | Adjacent safety/control reference; not a pure end-to-end policy. |
| 2023 | [Online RL with a safety supervisor](evans_et_al_2023_online_rl_supervisor.pdf) | Onboard real-world training protected by a supervisor. | Rare physical online-learning evidence. |
| 2023 | [Partial end-to-end reinforcement learning for autonomous racing](murdoch_et_al_2023_partial_end_to_end_rl.pdf) | Retains structure around the learned mapping rather than learning the whole stack. | Adjacent architectural comparison. |
| 2024 | [Offline reinforcement learning for autonomous racing](koirala_fleming_2024_offline_rl.pdf) | Trains policies from expert demonstration datasets and studies several policy representations. | Useful offline alternative when physical interaction is costly. |
| 2024 | [RACER: Epistemic risk-sensitive RL for small-scale racing](stachowicz_levine_2024_racer_safe_rl.pdf) | Risk-sensitive, uncertainty-aware racing policy. | Adjacent platform rather than F1TENTH proper. |
| 2025 | [RLPP: Reinforcement-learning pure pursuit](ghignone_et_al_2025_rlpp.pdf) | Learns residual corrections to PP and targets zero-shot real transfer. | Reports faster laps and a substantially reduced sim-to-real gap relative to its comparisons. |
| 2025 | [On learning racing policies with reinforcement learning](czechmanowski_et_al_2025_learning_racing_policies.pdf) | Actuator modeling and DR for aggressive zero-shot physical racing. | Direct, high-performance deployment study; compare claims within its protocol only. |
| 2025 | [Drive Fast, Learn Faster](hildisch_et_al_2025_onboard_rl.pdf) | Onboard SAC, including residual and end-to-end variants. | Physical learning in about 20 minutes; rare sample-efficiency result. |
| 2026 | [Learning to tune pure pursuit in autonomous racing](elgouhary_elwakeel_2026_tune_pure_pursuit.pdf) | PPO tunes PP lookahead and/or steering behavior. | Sim-to-real hybrid-control study. |
| 2026 | [Dynamic lookahead distance via RL-based pure pursuit](elgouhary_elwakeel_2026_dynamic_lookahead.pdf) | Learns state-dependent PP lookahead. | Closely related/overlapping author line; do not count as independent replication. |
| 2026 | [Efficient real-world racing via attenuated residual policy optimization](trumpp_et_al_2026_attenuated_residual_policy.pdf) | Progressively attenuates the base policy so the neural policy becomes standalone. | Direct response to residual-policy dependence; physical RoboRacer evidence. |
| 2026 | [Continual-RL for generalization on the RoboRacer platform](siegert_et_al_2026_continual_rl.pdf) | Continual backpropagation plus SAC and real-only fine-tuning. | Studies fast adaptation to new physical tracks. |

Across these studies, four design choices recur: a racing-line-relative observation/reward, a classical base controller, explicit safety supervision, and dynamics randomization/modeling. Their recurrence is meaningful: lap time alone is not enough to cross the simulation-to-hardware gap.

## Head-to-head racing

| Year | Paper | Learning setup | Evidence / caveat |
|---:|---|---|---|
| 2021 | [Offline learning of an overtaking policy for autonomous racing](bhargav_et_al_2021_offline_overtaking_policy.pdf) | Learns overtaking-probability maps offline and switches an MPCC strategy. | Early H2H hybrid; simulated, comparatively simple opponent model. |
| 2021 | [Stress testing autonomous racing overtake maneuvers](bak_et_al_2021_stress_testing_overtakes.pdf) | Adversarial/falsification analysis of overtaking. | Not a racing policy, but valuable for scenario generation and safety evaluation. |
| 2022 | [Game-theoretic objective-space planning](zheng_et_al_2022_game_theoretic_planning.pdf) | Explicit interactive game/planning rather than RL. | Important non-learning H2H baseline and potential expert generator. |
| 2023 | [Sense, Imagine, Act: Multimodal MBRL for head-to-head racing](shrestha_et_al_2023_sense_imagine_act.pdf) | Dreamer-style multimodal world model trained with static obstacles, evaluated zero-shot against a moving rule-based car. | Strong early learning-based H2H result; opponent diversity is limited. |
| 2023 | [Decentralized multi-agent deep RL for autonomous racing](samak_et_al_2023_multi_agent_drl.pdf) | Competitive decentralized MARL in a F1TENTH digital twin. | Core multi-agent RL reference; primarily simulation. |
| 2023 | [Machine-learning-based overtaking for autonomous racing](zhang_loidl_2023_ml_overtaking.pdf) | Learned/algorithmic overtaking components. | Adjacent H2H work; not standard end-to-end IL/RL. |
| 2024 | [MEGA-DAgger](sun_et_al_2024_mega_dagger.pdf) | H2H IL from multiple imperfect experts. | Includes physical overtaking and reports better collision/overtake metrics than its vanilla aggregation baseline. |
| 2024 | [RaceMOP: Mapless online path planning for multi-agent racing](trumpp_et_al_2024_racemop.pdf) | Residual RL over an artificial-potential-field policy using local observations. | Twelve simulated tracks; strong mapless collision-avoidance/generalization reference. |
| 2024/25 | [ForzaETH race stack](baumann_et_al_2024_forzaeth_stack.pdf) | Modular planning/control stack used in competitive head-to-head RoboRacer. | Not IL/RL, but a highly relevant physical systems baseline. |
| 2024 | [Data-driven aggressive racing control](li_et_al_2024_data_driven_aggressive_racing.pdf) | Data-driven dynamics/control for aggressive racing. | Adjacent learned-model/controller baseline. |
| 2025 | [FSDP: Fast and safe overtaking with data-driven prediction](hu_et_al_2025_fsdp.pdf) | Sparse-GP opponent prediction plus optimization. | Non-IL/RL baseline; reports improved success and lower computation in its protocol. |
| 2025 | [End2Race](end2race_2025.pdf) | Recurrent end-to-end LiDAR policy for both solo and interactive racing. | Broad simulated evaluation; reports 94.2% safety, 59.2% overtake rate, and sub-0.5 ms inference in its test suite. |
| 2025 | [Context-aware model-based RL for autonomous racing](moustafa_dusparic_2025_context_aware_mbrl.pdf) | Learns an opponent-context mask within a world model. | Directly studies generalization to unseen opponent behavior; simulation only. |
| 2025 | [Accelerating real-world overtaking in F1TENTH racing](steiner_et_al_2025_real_world_overtaking_rl.pdf) | Compares race-only and opponent-trained RL policies on hardware. | Particularly valuable physical comparison; reports 87% vs. 56% overtaking in its setup. |
| 2025 | [RL-based dynamic adaptation for sampling-based motion planning](langmann_et_al_2025_rl_dynamic_planner.pdf) | PPO adjusts planner cost weights during aggressive interactions. | Simulation; reports no collisions and shorter overtaking time than static-weight planners in its scenarios. |
| 2026 | [Autonomous overtaking trajectory optimization using RL and opponent pose estimation](cihlar_et_al_2026_overtaking_pose_estimation.pdf) | PPO racing policy plus camera/depth-based detection and UKF opponent estimation. | Physical F1TENTH overtakes; generalization degrades on different track geometry. |
| 2026 | [Physics-informed RL of spatial density velocity potentials](sivashangaran_et_al_2026_physics_informed_rl.pdf) | Map-free field representation used for time trials and overtaking. | Scaled hardware evidence; includes an author-reported OOD comparison to demonstrations. |

The H2H papers fall into three families: (a) **explicit interaction models** such as game theory, learned overtaking maps, and opponent prediction; (b) **reactive local policies** such as MEGA-DAgger, RaceMOP, and pose-conditioned PPO; and (c) **memory/world-model policies** such as Sense-Imagine-Act, context-aware MBRL, and End2Race. The last family is especially attractive when LiDAR geometry aliases the wall and opponent or when opponent intent must be inferred over time, but current evidence does not yet establish it as universally better.

## Foundations, surveys, and adjacent studies

| Year | Paper | Why retained |
|---:|---|---|
| 2019 | [F1/10: An open-source autonomous cyber-physical platform](okelly_et_al_2019_f1_10_platform.pdf) | Original platform/system reference. |
| 2020 | [F1TENTH: An open-source evaluation environment for continuous control and reinforcement learning](okelly_et_al_2020_f1tenth_rl_environment.pdf) | Canonical simulator/benchmark reference. |
| 2020 | [Sim-to-real learning for miniature autonomous racing](chu_et_al_2020_miniature_sim_to_real.pdf) | Broader scaled-racing sim-to-real reference. |
| 2021 | [Deep imitative reinforcement learning for autonomous racing](cai_et_al_2021_deep_imitative_rl.pdf) | Broader small-scale platform; useful hybrid IL/RL method. |
| 2024 | [A unifying F1TENTH autonomous racing survey](evans_et_al_2024_unifying_f1tenth_survey.pdf) | Best broad F1TENTH systems taxonomy and pre-2024 bibliography. |
| 2025 | [Advancing autonomous racing: A RoboRacer survey](wani_et_al_2025_roboracer_survey.pdf) | Recent platform survey; useful as a lead list, but individual citations should be checked against primary papers. |
| 2026 | [Neural-process reactive controller for autonomous racing](hunter_enyioha_2026_neural_process_controller.pdf) | Physics-informed/supervised controller with CBF in a F1TENTH-style simulator; not IL/RL. |
| 2026 | [Vision-based neural controllers with semi-probabilistic safety guarantees](ma_et_al_2026_vision_safety_guarantees.pdf) | Physical F1TENTH lane-following validation; safety-learning adjacent rather than racing-focused. |

## Papers not automatically downloadable

These are relevant enough to obtain manually. The metadata and stable landing links are provided so they can be located without another search.

1. **Michael Bosello, Rita Tse, and Giovanni Pau. “Train in Austria, Race in Montecarlo: Generalized RL for Cross-Track F1TENTH LIDAR-Based Races.”** *2022 IEEE 19th Annual Consumer Communications & Networking Conference (CCNC)*, pp. 290–298. DOI: [10.1109/CCNC49033.2022.9700730](https://doi.org/10.1109/CCNC49033.2022.9700730). LiDAR/DQN-style single-car policy and cross-track generalization; publisher and ResearchGate copies returned access errors.

2. **Tanay Dwivedi, Tobias Betz, Florian Sauerbeck, P. V. Manivannan, and Markus Lienkamp. “Continuous Control of Autonomous Vehicles using Plan-assisted Deep Reinforcement Learning.”** *2022 22nd International Conference on Control, Automation and Systems (ICCAS)*, pp. 244–250. DOI: [10.23919/ICCAS55662.2022.10003698](https://doi.org/10.23919/ICCAS55662.2022.10003698). Plan-conditioned Dreamer/world-model controller evaluated in simulation and on F1TENTH hardware; publisher copy was blocked.

3. **Ruiqi Zhang, Jing Hou, Guang Chen, Zhijun Li, Jianxiao Chen, and Alois Knoll. “Residual Policy Learning Facilitates Efficient Model-Free Autonomous Racing.”** *IEEE Robotics and Automation Letters* 7(4), 2022, pp. 11625–11632. DOI: [10.1109/LRA.2022.3192770](https://doi.org/10.1109/LRA.2022.3192770). The ResRace F1TENTH/PyBullet paper; the [TUM repository record](https://mediatum.ub.tum.de/1687707) is open in principle, but its anti-bot challenge prevented retrieval here.

4. **Máté Hell, Gergely Hajgató, Ármin Bogár-Németh, and Gergely Bári. “A LiDAR-Based Approach to Autonomous Racing with Model-Free Reinforcement Learning.”** *2024 IEEE Intelligent Vehicles Symposium (IV)*, pp. 258–263. DOI: [10.1109/IV55156.2024.10588613](https://doi.org/10.1109/IV55156.2024.10588613). Single-vehicle F1TENTH simulation study; IEEE download was blocked.

5. **Kai Yu, Mengyin Fu, Ting Zhang, and Yi Yang. “Enhancing Safety in Autonomous Racing With Constrained Reinforcement Learning.”** *IEEE Robotics and Automation Letters* 10(6), 2025, pp. 6448–6455. DOI: [10.1109/LRA.2025.3566591](https://doi.org/10.1109/LRA.2025.3566591). CMDP/constrained RL plus rollover protection, evaluated in F1TENTH simulation and on a 1:10 vehicle; no accessible author manuscript was found.

6. **Csanád Budai, Tamás Széles, Balázs Németh, and Péter Gáspár. “End-to-end Reinforcement Learning for Autonomous Racing: Bridging the Sim-to-Real Gap.”** *2025 American Control Conference (ACC)*, pp. 200–205. DOI: [10.23919/ACC63710.2025.11107738](https://doi.org/10.23919/ACC63710.2025.11107738). Domain-randomized end-to-end RL with a F1TENTH-type test vehicle; IEEE download was blocked.

7. **Elena Shrestha, Hanxi Wan, Chetan Reddy, Yulun Zhuang, and Ram Vasudevan. “Multimodal Model-Based Reinforcement Learning for Autonomous Racing.”** *Deployable RL @ RLC 2024*, published 1 June 2024. [OpenReview record](https://openreview.net/forum?id=wXjsMTZzxF). Introduces Lucid Dreamer, a LiDAR/RGB MBRL agent with simulated zero-shot H2H and real 1:10 sim-to-real tests. OpenReview exposed the record but returned HTTP 403 for the PDF. This appears to be a workshop follow-up/extension of the downloaded Sense-Imagine-Act manuscript, so treat results as related rather than independent replication.

## Gaps and recommended experimental protocol

The strongest open opportunity is a **real, interaction-aware, recurrent H2H policy evaluated across both unseen tracks and unseen opponent styles**. A defensible experiment should report:

- completion and lap time/speed for solo racing;
- collision rate, overtake success, time-to-pass, defensive success, and fault assignment for H2H;
- at least a non-learning planner, BC, interactive IL, MFRL, and a residual/hybrid policy where feasible;
- multiple opponents: fixed-line, blocking, reactive, and a learned opponent not used in training;
- in-distribution and unseen track geometries, dynamics/latency shifts, and randomized starting gaps;
- mean, dispersion/confidence intervals, number of independent seeds, and exact episode counts;
- inference latency and compute/power platform;
- simulation, zero-shot hardware, and hardware-fine-tuned results reported separately;
- safety interventions and near-collisions, not only completed laps.

For End2Race specifically, the closest comparisons are MEGA-DAgger (multi-expert IL and physical overtaking), RaceMOP (mapless residual RL), Sense-Imagine-Act/Lucid Dreamer and context-aware MBRL (memory/world models), Steiner et al. and Cihlar et al. (physical overtaking), and ForzaETH/GO(SP)/FSDP (strong non-end-to-end baselines). The most persuasive next step would be physical H2H evaluation with opponent-held-out and track-held-out splits rather than another same-track simulation comparison.

## Inventory note

Every local PDF linked above was checked for a valid PDF signature and readable metadata. The folder also retains methodologically adjacent material so future searches do not have to rediscover and re-screen it. Reported percentages in this review are the respective authors' results under their own protocols, not a cross-paper leaderboard.
