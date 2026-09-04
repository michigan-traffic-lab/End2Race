| Variant       | Single-Vehicle Speed (m/s) | Single-Vehicle Lap Time (s) | Head-to-Head Overtake (%) | Head-to-Head Safety (%) |
| ------------- | -------------------------: | --------------------------: | ------------------------: | ----------------------: |
| MLP           |                        6.9 |                        60.9 |                      67.6 |                    70.4 |
| Transformer   |                        6.7 |                        63.2 |                      82.8 |                    86.9 |
| Linear Norm.  |                        6.4 |                        64.9 |                      47.0 |                    73.3 |
| No Norm.      |                          - |                           - |                      16.3 |                    21.1 |
| Baseline (BC) |                        6.7 |                        62.7 |                      62.8 |                    82.8 |



The ablation study focuses on the two components that distinguish the End2Race policy from a conventional feedforward controller: temporal modeling and LiDAR proximity encoding. The GRU is introduced to carry information across consecutive observations, which is particularly important when the opponent cannot be fully characterized from a single LiDAR scan. We therefore replace it with two alternative architectures while leaving the sensory input and control head unchanged. The first is an MLP, which removes temporal memory and maps each 210-dimensional observation independently through a \(210 \rightarrow 420\) layer with a \(\tanh\) activation that keeps its output bounded like the recurrent state it replaces. The second is a causal Transformer with two encoder layers. The observation enters the first layer at its own width, which makes the model dimension 210; each layer uses three attention heads and a feed-forward dimension of 420, so the widening to 420 happens inside the feed-forward sublayer rather than on the input. Every step attends to at most the preceding 40 steps, one second at the 40 Hz control rate, and the model is trained on 40-step sliding windows so that each action is supervised with the same context the policy has online. The MLP feature is passed to the original \(420 \rightarrow 128 \rightarrow 2\) decoder, and the Transformer feature to a \(210 \rightarrow 128 \rightarrow 2\) decoder of the same depth. The replacements keep the input interface fixed, which centers the comparison on how temporal information is propagated; the MLP, the Transformer, and End2Race hold \(0.14\), \(0.74\), and \(0.85\) million trainable parameters respectively.

The GRU architecture is then retained to examine the effect of the LiDAR representation independently. End2Race converts each measurement \(x\) using \(2/(1+e^{0.3x})\), so nearby objects receive larger values and distant measurements gradually approach zero. For comparison, all LiDAR measurements are first truncated to \(\min(x,10)\). We train one GRU with linear normalization, \(\min(x,10)/10\), and another directly on the truncated distance \(\min(x,10)\). Only the LiDAR preprocessing is changed for these two models. The speed embedding, recurrent state, masking procedure, and action decoder follow the original policy. Each architecture is then trained from the same collision-free Austin demonstrations using the BC procedure described in Sec.\ref{sec:behavioral-cloning}. We use Adam with a learning rate of \(10^{-4}\) for 500 epochs and a batch size of 1024. The Transformer is the one exception, because its training set is the 180{,}683 forty-step windows extracted from those demonstrations rather than the 643 episodes: a single epoch already contains 177 updates, so its 500 epochs amount to 88{,}500 parameter updates against the 500 the other models receive. Each window supervises only its final action, except for the window that opens an episode, which supervises all 40 of its positions, so every expert action is supervised exactly once per epoch, and every model therefore sees the same demonstrations the same number of times. The loss remains \(\mathcal{L}=\mathcal{L}_{\mathrm{steer}}+0.05\mathcal{L}_{\mathrm{speed}}\), and the ego-speed embedding is replaced by the learned mask embedding at 20% of training timesteps. Gradient clipping is applied with a maximum norm of 1.0.

Because these experiments are intended to isolate model design rather than cross-track generalization, all ablations are evaluated on Austin. Single-vehicle performance is measured over one complete lap using mean speed and lap time. Head-to-head performance uses the same 720 overtaking scenarios introduced in Sec.\ref{sec:overtaking-setup}: 80 starting positions are combined with three opponent racelines and speed scales of 0.4, 0.6, and 0.8, with each scenario lasting 8 seconds. The vehicle initialization, opponent policy, and simulator configuration are kept consistent across the models. A scenario is counted as an overtake when the ego vehicle finishes ahead of the opponent. Both successful overtakes and collision-free following cases contribute to the safety rate. Every ablated architecture is trained with three random seeds, and the head-to-head numbers are averaged over them. The reported single-vehicle speed and lap time average only the seeds that complete the lap, which is all three for the MLP and the Transformer, two of three for linear normalization, and none for the unnormalized model, whose single-vehicle entries are therefore left empty. The corresponding single-vehicle and head-to-head measurements are summarized in Table\ref{tab:ablation}.

---

## Run record

Twelve models: four ablations, three seeds each, all trained on `dataset/success` (643 Austin
episodes of 320 steps) and evaluated on Austin with seed 42. Every one of the twelve head-to-head
evaluations reported `errors: 0` over the full 720 planned scenarios.

### Per seed

| Variant | seed | final loss | overtake % | safety % | collisions /720 | single lap | lap time (s) | speed (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| Transformer | 1 | 0.00066 | 83.9 | 87.4 | 91 | pass | 63.32 | 6.64 |
| Transformer | 2 | 0.00058 | 82.9 | 87.4 | 91 | pass | 63.19 | 6.68 |
| Transformer | 3 | 0.00051 | 81.7 | 86.1 | 100 | pass | 63.20 | 6.67 |
| MLP | 1 | 0.01542 | 69.0 | 72.6 | 197 | pass | 60.72 | 6.89 |
| MLP | 2 | 0.01599 | 67.1 | 68.9 | 224 | pass | 60.97 | 6.88 |
| MLP | 3 | 0.01643 | 66.7 | 69.7 | 218 | pass | 60.98 | 6.86 |
| Linear Norm. | 1 | 0.01083 | 54.7 | 80.3 | 142 | pass | 63.76 | 6.56 |
| Linear Norm. | 2 | 0.01026 | 45.3 | 69.9 | 217 | off track at 9.7 % | - | - |
| Linear Norm. | 3 | 0.01150 | 41.0 | 69.9 | 217 | pass | 66.02 | 6.32 |
| No Norm. | 1 | 0.00808 | 31.8 | 36.0 | 461 | off track at 21.9 % | - | - |
| No Norm. | 2 | 0.00780 | 12.8 | 17.9 | 591 | off track at 2.9 % | - | - |
| No Norm. | 3 | 0.01027 | 4.3 | 9.3 | 653 | off track at 2.5 % | - | - |

### Seed spread

| Variant | mean loss | overtake % | SD | safety % | SD | laps completed |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Transformer | 0.00058 | 82.8 | 1.1 | 86.9 | 0.7 | 3/3 |
| MLP | 0.01595 | 67.6 | 1.3 | 70.4 | 2.0 | 3/3 |
| Linear Norm. | 0.01086 | 47.0 | 7.0 | 73.3 | 6.0 | 2/3 |
| No Norm. | 0.00871 | 16.3 | 14.1 | 21.1 | 13.6 | 0/3 |
| Baseline (BC) | - | 62.8 | - | 82.8 | - | - |

### What the numbers show

The Transformer is the strongest model on both head-to-head measures, ahead of the recurrent
baseline by 20 points of overtake rate and 4 points of safety, and it is also the most reproducible:
its overtake rate varies by 1.1 points across seeds, against 14.1 for the unnormalized GRU. Its
advantage is a safety advantage rather than a speed advantage, since it collides in 91 to 100 of the
720 scenarios where the MLP collides in 197 to 224.

That comparison is not a clean architecture comparison. Because the Transformer trains on windows
rather than on whole episodes, its 500 epochs deliver 88,500 parameter updates while the other three
models receive 500, and its training loss is an order of magnitude lower as a result. The optimizer
budget and the architecture move together here, so the measured gap should not be read as an
architectural effect on its own. Separating them needs a Transformer held to 500 updates.

Removing temporal memory costs safety rather than pace. The MLP laps fastest of every model at
60.9 seconds and completes the lap on all three seeds, yet it collides more than twice as often as
the Transformer and lands 12 points of safety below the recurrent baseline. A single LiDAR scan is
evidently enough to drive the line but not enough to negotiate an opponent.

The LiDAR encoding results are monotone and need no such caveat, since those three models share the
GRU, the training procedure, and the update budget. Replacing \(2/(1+e^{0.3x})\) with linear
normalization drops the overtake rate from 62.8 % to 47.0 % and widens the seed spread from a
baseline that is stable to an SD of 7.0; dropping normalization altogether drops it to 16.3 % with
an SD of 14.1 and no seed that finishes a lap. Compressing far measurements toward zero, so that
LiDAR returns read as obstacle pressure rather than as distance, is what keeps the policy stable.

### Artifacts

```
checkpoint/ablation_{transformer,mlp,linear,nonorm}/run_{1,2,3}/
    epoch_00500.pt
    metrics.csv                                  # epoch,total_loss,steering_loss,weighted_speed_loss
eval_results/ablation_<variant>/run_{1,2,3}/
    single/results.json
    multi/epoch_00500/Austin/results.json
```

Training entry points are `imitation/train_transformer.py`, `imitation/train_mlp.py`, and
`imitation/train_preprocess.py --preprocessing {linear,no_normalization}`. Evaluation entry points
are `evaluation/evalsingle_ablation_{transformer,mlp,preprocess}.py` and the matching
`evaluation/evalmulti_ablation_*.sh`, run with `WORKERS=12` on Austin.
