import gymnasium as gym
from stable_baselines3 import PPO
from environment.chess_env import ChessEnv

# Load trained model
model = PPO.load('chess_agent_PPO_V0')


# Initialize environment
env = ChessEnv()

observation, info = env.reset()

# Run a single episode
done = False
while not done:
    action, _states = model.predict(observation)
    obs, reward, done, _, info = env.step(action)
    env.render()


env.close()