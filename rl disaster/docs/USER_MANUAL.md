# RL Disaster Evacuation System - User & Operator Manual

Welcome to the **RL Disaster Evacuation System** User Manual. This guide provides clear, practical instructions on how to install, launch, operate, and evaluate the evacuation simulation.

---

## 1. Quick Start Guide

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10 or higher
- **Display**: Minimum screen resolution of $1280 \times 720$

### Starting the Application in 2 Steps

1. Open PowerShell or your terminal in the project folder:
   ```powershell
   cd "c:\Users\rasto\Downloads\rl disaster"
   ```

2. Run the interactive visual simulator:
   ```powershell
   .\.venv\Scripts\python.exe demo_pygame.py
   ```

Within a few seconds, the simulation window will launch displaying the 2D city map, starting zone, safe exit, and real-time dashboard.

---

## 2. Interactive Map & Display Overview

The simulation renders a stylized top-down city environment:

```
+------------------------------------------------------------------------+
|                                    [SAFE EXIT (Green Zone)]            |
|   [Parks & Trees]       [City Streets]               |                 |
|                                                      v                 |
|   [Buildings]           +--- Roadblock (Orange Cone)                   |
|                         |                                              |
|                         +--- Evacuation Path (Cyan / Green / Blue)     |
|                                                                        |
| [START ZONE (Orange)]                                                  |
+------------------------------------------------------------------------+
| HUD Telemetry: Algorithm | Step | Reward | Safe Exits | Active Hazards |
+------------------------------------------------------------------------+
```

### Visual Elements Guide:
- **Green Grassy Areas & Trees**: Open parks, non-traversable for vehicles.
- **Dark Grey Corridors**: City streets. Streets are strictly **one lane wide**, requiring smart detours when blocked.
- **Navy & Slate Blocks**: Residential and commercial buildings (solid barriers).
- **Orange Circular Marker (Bottom-Left)**: Starting evacuation zone `(21, 1)`.
- **Green Glowing Zone (Top-Right)**: Safe evacuation exit `(3, 32)`.
- **Orange Traffic Cones**: Roadblocks caused by accidents or debris.
- **Red / Orange Fire Zones**: Dynamic spreading fires that must be avoided.
- **Colored Path Lines**:
  - **Blue Line**: Route planned by **Dijkstra's Algorithm**.
  - **Green Line**: Route planned by **A\* Search Algorithm**.
  - **Cyan Dot & Trail**: Live position and path taken by the **Trained RL (PPO) Agent**.

---

## 3. Keyboard Controls & Hotkeys

All interactive commands can be triggered using your keyboard during simulation:

| Key | Action | Description |
| :---: | :--- | :--- |
| **`1`** | **Select Dijkstra** | Activates the classic shortest-path algorithm. Recalculates detours around hazards. |
| **`2`** | **Select A\* Search** | Activates heuristic-guided A\* search for faster, direct path planning. |
| **`3`** | **Select PPO Agent** | Activates the trained AI neural network agent to navigate dynamically. |
| **`SPACE`** | **Pause / Resume** | Freezes the simulation clock to inspect paths and statistics, or resumes play. |
| **`R`** | **Reset Episode** | Places the agent back at the starting zone with a fresh scenario seed. |
| **`B`** | **Add Roadblock** | Places a dynamic obstruction on the road to test how algorithms respond. |
| **`C`** | **Toggle Crowds** | Shows or hides pedestrian crowd density heatmaps on the streets. |
| **`M`** | **Cycle Scenario** | Switches between: *Normal*, *Random Roadblocks*, *Fire Spread*, and *Full Crisis*. |
| **`+` / `-`** | **Adjust Speed** | Speeds up or slows down simulation playback speed. |
| **`ESC`** | **Quit** | Closes the simulation cleanly. |

---

## 4. Understanding the Evaluation Scenarios

The simulator allows testing four emergency scenarios:

1. **Scenario 1: Clear Day (Baseline)**
   - All roads are open.
   - Purpose: Validates that all algorithms find the most direct route to the exit without hesitation.

2. **Scenario 2: Dynamic Roadblocks**
   - Accidents and fallen trees randomly block streets while the agent is in transit.
   - Purpose: Tests whether the algorithms dynamically recalculate detours without getting stuck or cutting through buildings.

3. **Scenario 3: Spreading Fire Hazard**
   - A fire starts at a designated point and progressively spreads to neighboring street tiles.
   - Purpose: Tests safety buffers and hazard avoidance behavior.

4. **Scenario 4: Congested Evacuation (Crowd + Blockages)**
   - High pedestrian density causes traffic slowdowns alongside blocked roads.
   - Purpose: Tests the AI's ability to trade off distance against congestion delays.

---

## 5. How the AI Learns: The Reward Function Explained

The AI agent does not have human intuition; instead, it learns how to evacuate safely through a **Reward and Penalty System** (similar to points in a game). Every time the agent makes a move, the environment gives it points or takes points away based on how safe and effective that move was.

### The Reward Formula

$$\text{Reward} = R_{\text{step}} + R_{\text{progress}} + R_{\text{hazard}} + R_{\text{crowd}} + R_{\text{terminal}}$$

### Breakdown of Reward Rules:

1. **Reaching Safety ($+100.0$ points)**
   - When the agent enters the safe exit zone, it receives a large positive reward. This is the primary objective of the mission.

2. **Making Forward Progress ($+1.5 \times \text{distance closer}$)**
   - Every time the agent moves one step closer to the exit, it earns positive points. If it moves backward away from the exit, it is penalized.

3. **Time Clock Penalty ($-0.1$ points per step)**
   - A tiny deduction is applied for every tick of time. This teaches the agent not to wander aimlessly or wait around during a disaster.

4. **Collision with Walls or Buildings ($-5.0$ points)**
   - If the agent attempts to step into a solid building or off the map boundary, it bumps into the obstacle and receives a sharp penalty.

5. **Fire and Smoke Hazards ($-10.0$ to $-20.0$ points)**
   - Entering an active fire zone or heavy smoke incurs heavy penalties, training the agent to give hazards a wide berth.

6. **Avoiding Congestion ($-0.5 \times \text{crowd density}$)**
   - Walking through dense, slow-moving crowds slows the agent down and slightly lowers the reward, encouraging it to take less crowded parallel streets when available.

7. **Anti-Loitering / Re-visiting Penalty ($-1.0$ point)**
   - Stepping back onto a tile it just visited applies a small penalty. This prevents the agent from getting stuck in an infinite two-tile loop.

Through thousands of practice episodes, the AI naturally discovers routes that are both **fast** and **safe**.

---

## 6. Frequently Asked Questions (FAQ)

- **Q: Why does the path change color when I press 1, 2, or 3?**
  - **A**: Each number activates a different navigation brain (Blue = Dijkstra, Green = A*, Cyan = Trained PPO AI). The color allows you to visually compare their decisions side-by-side.
- **Q: What happens if a roadblock completely blocks the only road to the exit?**
  - **A**: The algorithms will detect that no valid path exists and display a "No Path Found" alert on the HUD without crashing.
- **Q: Can I add more roadblocks while the agent is walking?**
  - **A**: Yes! Press `B` at any time while the simulation is running to trigger a roadblock and watch the agent immediately re-route.

---

## 7. Appendix: Technical Terms & Glossary

For readers interested in the technical terminology used throughout the system:

- **Agent**: The autonomous decision-maker (in this project, the simulated evacuee navigating the city).
- **A\* (A-Star) Search**: A popular graph traversal algorithm that uses an educated guess (heuristic) of remaining distance to find the shortest path faster than Dijkstra.
- **Action Space**: The set of all possible moves available to the agent at any moment (Up, Down, Left, Right, or Stay).
- **Convergence**: The stage in machine learning training where the model has learned the optimal strategy and its performance stabilizes at peak success.
- **Dijkstra's Algorithm**: A foundational mathematical method that calculates the shortest path from a starting point to all other reachable nodes on a map.
- **Dynamic Hazard**: An obstacle or danger (such as an expanding fire or sudden roadblock) that changes location or intensity over time.
- **Episode**: One complete simulation run from the starting point until the agent reaches the exit, gets trapped, or runs out of allowed time.
- **Heuristic**: A practical estimate or rule of thumb used by algorithms to guide decision-making without exploring every single possibility.
- **Hyperparameter**: Settings chosen by engineers prior to training (such as learning rate, batch size, or discount factor) that control how the AI learns.
- **Manhattan Distance**: The distance between two points measured along right-angle grid lines (like city blocks), calculated as $|x_1 - x_2| + |y_1 - y_2|$.
- **Observation Space**: The data provided to the AI agent representing what it can see around itself (its own coordinates, exit direction, and a $5 \times 5$ local view of roads and hazards).
- **Policy ($\pi$)**: The learned strategy inside the neural network that maps what the agent sees (observations) to what action it should take.
- **PPO (Proximal Policy Optimization)**: A state-of-the-art Reinforcement Learning algorithm known for stable, reliable learning by preventing overly large policy updates.
- **Reinforcement Learning (RL)**: A branch of Artificial Intelligence where agents learn optimal behavior through trial and error, guided by rewards and penalties.
- **Single-Lane Corridor**: A map design where road sections are exactly one tile wide, ensuring that obstacles cannot simply be sidestepped on the same road block.
- **State**: A complete snapshot of the simulation at a specific split second (agent location, open roads, fire locations, crowd positions).
