from twip_env import TwipEnv
from stable_baselines3 import PPO

env = TwipEnv()
episodes = 500

model = PPO('MlpPolicy', env, device="cpu")
for episode in range(episodes):
    done = False
    obs, _ = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rew, dones, truncated, info = env.step(action)