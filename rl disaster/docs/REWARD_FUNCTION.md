# Reward Function Documentation — v2

> Updated to reflect anti-oscillation penalties and topological direction features added in the v2 training run.

## 1. Overview

The PPO agent receives a scalar reward signal $r_t$ at each discrete time step $t$. The total episode return is:

$$R = \sum_{t=0}^{T} \gamma^t \, r_t, \quad \gamma = 0.995$$

The reward is composed of seven additive terms:

$$r_t = r_{\text{step}} + r_{\text{wall}} + r_{\text{progress}} + r_{\text{detour}} + r_{\text{hazard}} + r_{\text{congestion}} + r_{\text{anti-osc}} + r_{\text{goal}}$$

---

## 2. Component Breakdown

### 2.1 Step Penalty — $r_{\text{step}}$

$$r_{\text{step}} = -0.5 \quad \text{(every step)}$$

Discourages unnecessary idling; motivates fast evacuation.

---

### 2.2 Wall / Blockage Collision Penalty — $r_{\text{wall}}$

| Obstacle type | Penalty |
|---|---|
| Building wall / perimeter | $-2.0$ |
| Barricade / rubble / car wreck | $-25.0$ |

Trying to walk through a blocked cell incurs this penalty **and** the agent does **not** move.

---

### 2.3 Geodesic Progress Shaping — $r_{\text{progress}}$

This is the key signal for learning to navigate around obstacles:

$$r_{\text{progress}} = w_p \cdot \text{clip}\!\left(\Phi(s_t) - \Phi(s_{t+1}),\ -2,\ +2\right)$$

where:
- $w_p = 2.5$ (progress reward weight)
- $\Phi(s)$ = **topological geodesic BFS distance** from state $s$ to the nearest open exit, computed over the passable road graph (blocked cells excluded)

> [!IMPORTANT]
> Unlike Euclidean/Manhattan distance, the geodesic distance through the road graph automatically captures detour routes. When a direct path is blocked, moving along a longer detour **still yields positive progress** because the BFS distance decreases along the alternative route.

---

### 2.4 Detour Bonus — $r_{\text{detour}}$ *(new in v2)*

$$r_{\text{detour}} = 1.5 \cdot \min\!\left(\Phi(s_t) - \Phi(s_{t+1}),\ 1.0\right) \quad \text{if } \Phi(s_t) > \Phi(s_{t+1})$$

An extra bonus on top of the progress shaping that fires whenever the agent **actually gets closer** to the exit. This doubles the incentive to commit to detours even when they temporarily look counter-intuitive to the network.

---

### 2.5 Hazard Penalty — $r_{\text{hazard}}$

| Situation | Penalty |
|---|---|
| Entering fire / explosion / blocked cell | $-75.0$ (terminal) |
| Passing through smoke / dust cloud | $-6.0$ |

---

### 2.6 Congestion Penalty — $r_{\text{congestion}}$

$$r_{\text{congestion}} = -3.0 \cdot \text{CrowdDensity}(s_{t+1})$$

Penalises routing through high-traffic cells (crowd density $\in [0, 1]$).

---

### 2.7 Anti-Oscillation Penalty — $r_{\text{anti-osc}}$ *(new in v2)*

Two-tier penalty system to eliminate the back-and-forth oscillation seen in v1:

| Trigger | Penalty |
|---|---|
| Agent returns to the cell it occupied **2 steps ago** (direct reversal) | $-3.0$ |
| Agent moves to any cell visited within the **last 6 steps** | $-1.5$ |

This directly penalises the oscillation pattern (A→B→A→B→…) during training. Combined with the topological direction vector in the observation, the policy network learns to commit to detours instead of reversing.

---

### 2.8 Goal Reward — $r_{\text{goal}}$

$$r_{\text{goal}} = 150.0 - 15.0 \cdot \text{ExitCongestion}$$

Awarded once when the agent reaches any open exit. Exits with heavy crowd congestion yield a slightly lower reward.

---

## 3. Observation Space (v3 — Single Distant Exit) — 112 Dimensions

| Component | Dims | Description |
|---|---|---|
| Agent position | 2 | `(row/H, col/W)` normalised to `[0,1]` |
| Exit features | 5 | 1 primary exit at `(0, 2)` × `(Δrow, Δcol, dist, blocked, congestion)` |
| **Topological direction** | **4** | `(topo_dr, topo_dc, geo_dist_norm, stuck_flag)` |
| Local 5×5 window | 100 | 4 channels: walls, car-blasts, rubble/barricade, crowd |
| Time | 1 | `step / max_steps` |
| **Total** | **112** | Complete input vector to PPO policy network |

### Spawn Distance Constraint
To prevent trivial shortcuts and ensure the agent learns multi-block cross-city evacuation:
$$\text{Manhattan Distance}(\text{Agent}_{\text{spawn}},\, \text{Exit}) \ge 15$$
Spawns are located across the distant sectors (Bottom-Right $B_9$ Danger Zone ~42 steps away, Bottom-Center $B_8$, Bottom-Left $B_7$, Mid-Right $B_6$, Top-Right $B_3$).

> [!NOTE]
> **Topological direction vector** is the single most important fix for the oscillation bug.
> `(topo_dr, topo_dc)` = direction of the **first step** on the BFS-optimal road-graph path to the nearest open exit.
> When a direct path is blocked, this vector automatically points along the detour (e.g., south and around a building), giving the neural net an explicit compass even in complex blockage scenarios.

---

## 4. Training Configuration — v2

| Parameter | Value |
|---|---|
| Total timesteps | **150,000** |
| Network architecture | MLP 256 → 256 → 128 (π & V) |
| Learning rate | 2.5 × 10⁻⁴ |
| n_steps per rollout | 2048 |
| Batch size | 128 |
| Discount factor γ | 0.995 |
| Entropy coefficient | 0.005 |
| Curriculum warmup | 40,000 steps |
| Hazard injection interval | every 600 steps (post-warmup) |

### Curriculum Schedule

```
Steps 0 – 40,000    : Pure routing (no mid-episode hazards)
Steps 40,000 – 150,000 : Random barricade / car-blast / building-collapse
                          injected every 600 steps to force active rerouting
```

> [!TIP]
> The curriculum ensures the agent first learns **basic street navigation** before being exposed to dynamic obstacles. Without warmup, early hazards often trap the agent before it learns to move at all.

---

## 5. Fixes Applied vs. v1

| Issue (v1) | Fix (v2) |
|---|---|
| Agent oscillates A↔B near blocked corner | Anti-oscillation penalty (`-3.0` / `-1.5`) in env |
| Policy driven by Euclidean exit vector into walls | Topological direction vector replaces raw Euclidean offset |
| Demo: PPO reversal loop not prevented | Visited-cell memory + reversal guard in `demo_pygame.py` |
| Only 30–60k training steps | 150,000 steps with curriculum |
| No mid-episode hazards during training | Dynamic curriculum from step 40,000 |
