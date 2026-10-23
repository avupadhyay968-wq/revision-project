import gymnasium as gym
from stable_baselines3 import PPO

# 1. Environment render mode ke saath create karein (Window popup hogi)
env = gym.make("CartPole-v1", render_mode="human")

# 2. Saved model load karein
model = PPO.load("cartpole_ppo_model")
print("Model loaded successfully!")

# 3. Test episodes run karein
NUM_EPISODES = 10

for episode in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Model observation dekh kar best action predict karta hai
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode} Score (Frames Balanced): {total_reward}")

env.close()