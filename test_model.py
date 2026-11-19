from src.env.evacuation_env import DisasterEvacuationEnv
from stable_baselines3 import PPO

env = DisasterEvacuationEnv()
model = PPO.load("models/ppo_evac.zip")
print("Model loaded OK, obs_dim:", env.observation_space.shape[0])
print("Exits:", env.grid_map.exits)

wins = 0
for i in range(20):
    obs, _ = env.reset()
    done = False
    steps = 0
    start = env.agent_pos
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
    evac = info.get("evacuated", False)
    if evac:
        wins += 1
    print(f"  Ep {i+1:2d}: spawn={start} steps={steps:3d} evacuated={evac}")
print(f"Success: {wins}/20 ({wins*5}%)")
