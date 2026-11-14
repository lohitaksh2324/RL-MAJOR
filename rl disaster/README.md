# RL-Based Disaster Evacuation System

An adaptive, safe, and congestion-aware evacuation simulation system powered by **Deep Reinforcement Learning (Proximal Policy Optimization - PPO)**. The system models dynamic disaster progression (fire expansion, smoke diffusion, crowd bottlenecks, blocked routes) and trains autonomous agents to navigate safely to emergency exits.

---

## 📌 Executive Summary

Traditional evacuation planning typically relies on static shortest-path algorithms such as **A\*** and **Dijkstra**. In real-world disasters, these approaches fail when:
1. **Dynamic Hazards Evolve:** Rapidly spreading fire and toxic smoke cut off the primary shortest path.
2. **Crowd Congestion Occurs:** Sending all occupants to the nearest exit creates severe bottlenecks.
3. **Information is Dynamic:** Safe routes degrade over time, demanding real-time adaptive rerouting.

This system formulates evacuation as a sequential decision-making problem (MDP) solved via **PPO**, optimizing for evacuation speed, hazard avoidance, and crowd congestion mitigation.

---

## 🏛️ System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Disaster Evacuation Env             │
                  │  (20x20 Grid: Rooms, Obstacles, 3 Exits)     │
                  └──────────────┬───────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────────────┐
         ▼                       ▼                               ▼
┌──────────────────┐   ┌──────────────────┐            ┌──────────────────┐
│  Disaster Model  │   │   Crowd Model    │            │   PPO Policy     │
│  - Fire Spread   │   │ - Exit Bottlenecks│           │  - Actor-Critic  │
│  - Smoke Density │   │ - Crowd Density  │            │  - Observation:  │
│  - Blocked Paths │   │ - Delay Penalties│            │    Local + Exits │
└──────────────────┘   └──────────────────┘            └──────────────────┘
         │                       │                               │
         └───────────────────────┼───────────────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │     Evaluation Framework     │
                  │  (Success, Steps, Exposure,  │
                  │   Congestion, Baselines)     │
                  └──────────────────────────────┘
```

The system comprises 5 core modules:
1. **Environment (`src/env/`):** 20x20 Gymnasium grid with realistic rooms, corridors, static debris, and 3 emergency exits.
2. **Disaster Model (`src/env/disaster_model.py`):** Dynamic cellular automaton fire spreading with probabilistic ignition and smoke dispersion.
3. **Crowd Model (`src/env/crowd_model.py`):** Simulates exit bottleneck queuing and crowd densities.
4. **Primary RL Agent (`src/rl/train_ppo.py`):** PPO agent trained with potential-based progress rewards and safety penalties.
5. **Baselines (`src/baselines/`):** A\* and Dijkstra algorithms evaluated under both static and dynamic replanning regimes.

---

## 🧠 Reinforcement Learning Formulation

- **State Space (118-dim vector):**
  - Normalized agent position $(r, c)$.
  - Relative displacement, normalized distance, fire blockage flag, and crowd congestion level for each of the 3 exits.
  - $5 \times 5$ local perception window around agent (4 channels: wall, fire, smoke, crowd density).
  - Normalized elapsed episode time.
- **Action Space:** Discrete(5) — `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`.
- **Reward Shaping:**
  - Step penalty: $-0.5$ (encourages fast evacuation).
  - Progress shaping: $+2.0 \times \Delta \text{distance}$ toward nearest reachable safe exit.
  - Fire contact penalty: $-60.0$ (fatal termination).
  - Smoke inhalation penalty: $-6.0$.
  - Static hazard penalty: $-25.0$.
  - Crowd congestion penalty: $-3.0 \times \text{density}$.
  - Successful exit reward: $+120.0$.

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed. Activate the project virtual environment:

```powershell
# Activate virtual environment
.\.venv\Scripts\activate
```

Dependencies are documented in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## 💻 Usage

### Diagnostic Component Tests
Run diagnostic verification of all environment and planning modules:
```powershell
python main.py --test
```

### Train Primary PPO Agent
Train the PPO agent and generate training curve plots (`results/ppo_training_curves.png`):
```powershell
python main.py --train --timesteps 45000
```

### Run Benchmark Comparison (PPO vs A* vs Dijkstra)
Benchmark across 4 challenging disaster scenarios (Standard, Blocked Route, Heavy Congestion, Compound Disaster):
```powershell
python main.py --benchmark --episodes 30
```
Outputs:
- JSON report: `results/benchmark_results.json`
- Comparison charts: `results/benchmark_comparison.png`

### Interactive 2D Pygame Simulator
Launch the real-time interactive graphical interface:
```powershell
python demo_pygame.py
```
**Keyboard Controls:**
- `[SPACE]`: Pause / Resume simulation
- `[R]`: Reset episode
- `[1]`: Switch to PPO (Adaptive RL)
- `[2]`: Switch to A\* (Static)
- `[3]`: Switch to A\* (Dynamic Replanning)
- `[4]`: Switch to Dijkstra
- `[S]`: Cycle disaster scenarios
- `[ESC]`: Exit

### Interactive Web Dashboard (Streamlit)
Launch the interactive web-based analytics dashboard:
```powershell
streamlit run src/visualization/streamlit_app.py
```

---

## 📊 Benchmark Scenarios

| Scenario ID | Name | Dynamic Disaster Challenge |
| :--- | :--- | :--- |
| **Scenario 1** | Standard Evacuation | Baseline multi-exit layout with distant fire. |
| **Scenario 2** | Dynamic Route Blockage | Fire abruptly flashes and blocks nearest exit at step 4, forcing adaptive rerouting. |
| **Scenario 3** | Heavy Exit Congestion | Nearest exit is 95% jammed by crowd bottleneck; farther exits are clear. |
| **Scenario 4** | Compound Disaster | High fire spread rate combined with multiple bottlenecked exits. |

---

## 📁 Repository Structure

```
rl disaster/
├── src/
│   ├── config.py                 # Hyperparameters, cell types, reward constants
│   ├── env/
│   │   ├── grid_map.py           # 2D floorplan layout, rooms, exits
│   │   ├── disaster_model.py     # Cellular automaton fire & smoke propagation
│   │   ├── crowd_model.py        # Crowd density & bottleneck queuing
│   │   └── evacuation_env.py     # Gymnasium MDP environment
│   ├── baselines/
│   │   ├── astar.py              # A* search (static & dynamic replanning)
│   │   └── dijkstra.py           # Dijkstra search (static & dynamic replanning)
│   ├── rl/
│   │   └── train_ppo.py          # Stable-Baselines3 PPO training & evaluation
│   ├── evaluation/
│   │   ├── metrics.py            # KPI dataclasses and aggregation
│   │   ├── benchmark.py          # Comparative evaluation across 4 scenarios
│   │   └── plotting.py           # Matplotlib figures generator
│   └── visualization/
│       ├── pygame_renderer.py    # Pygame 2D real-time simulation view
│       └── streamlit_app.py      # Streamlit browser interface
├── models/                       # Saved trained models (.zip)
├── results/                      # Benchmarks, logs, and generated charts
├── demo_pygame.py                # Standalone Pygame visualizer
├── main.py                       # CLI interface
├── requirements.txt              # Python package dependencies
└── README.md                     # Technical documentation
```
