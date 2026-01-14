from twip_env import TwipEnv
from stable_baselines3 import PPO

num_envs = 128
num_steps_per_env = 8192

env = TwipEnv(num_envs=num_envs)
model = PPO('MlpPolicy', env, device="cpu", tensorboard_log="./tb", n_steps=128, batch_size=256, n_epochs=4)

def train():
    try:
        model.learn(total_timesteps=num_steps_per_env * num_envs, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save("model_interrupt")

def evaluate():
    obs, _ = env.reset()
    for episode in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rew, dones, info = env.step(action)
            done = dones.any()  # Adjusted for vectorized envs

            print(f"Episode: {episode + 1}, Reward: {rew.sum().item()}")

if __name__ == "__main__":
    train()
    evaluate()

    model.save("model")