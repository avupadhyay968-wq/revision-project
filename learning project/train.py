import gymnasium as gym
from stable_baselines3 import PPO

# 1. Environment create karein
env = gym.make("CartPole-v1")

# 2. PPO (Proximal Policy Optimization) Model initialize karein
# MlpPolicy: Multi-Layer Perceptron (Standard feed-forward neural network)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.99
)

# 3. Model Train karein (Timesteps = total interactions)
print("--- Training RL Model ---")
model.learn(total_timesteps=20000)

# 4. Model save karein (Jaise ML models save hote hain)
model_path = "cartpole_ppo_model"
model.save(model_path)
print(f"\nModel successfully saved as: {model_path}.zip")

env.close()