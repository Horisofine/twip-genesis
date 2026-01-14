import os

from twip_env import TwipEnv
from stable_baselines3 import PPO

num_envs = 128
num_steps_per_env = 8192

env = TwipEnv(num_envs=num_envs)
model = PPO('MlpPolicy', env, device="cpu", tensorboard_log="./tb", n_steps=128, batch_size=256, n_epochs=6)
model_save_path = os.path.join(os.getcwd(), "model")

def train():
    model.learn(total_timesteps=num_steps_per_env * num_envs, progress_bar=True)

def evaluate():
    for episode in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rew, dones, info = env.step(action)
            done = dones.any()  # Adjusted for vectorized envs

            print(f"Episode: {episode + 1}, Reward: {rew.sum().item()}")

if __name__ == "__main__":
    from genesis import GenesisException

    try:
        train()
        evaluate()
    except KeyboardInterrupt:
        print("Training interrupted.")
    except GenesisException as e:
        print(f"Genesis Exception occurred: {e}.")
    finally:
        print("Saving model...")

        model.save(model_save_path)
