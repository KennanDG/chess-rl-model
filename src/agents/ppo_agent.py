from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.chess_env import ChessEnv

# Initialize Chess Evironment
env = DummyVecEnv([lambda: ChessEnv()])

# Initialize PPO agent
model = PPO(
    policy="MlpPolicy",  
    env=env,
    learning_rate=0.0002,  # Tuned learning rate (lower for stability)
    n_steps=8192,  # Larger rollout buffer for long-term strategies
    batch_size=512,  # Large batch size for better updates
    n_epochs=10,  
    gamma=0.99,  
    gae_lambda=0.95,  
    clip_range=0.1,  # Smaller clip range to prevent unstable updates
    ent_coef=0.01,  # Small entropy bonus to encourage exploration
    vf_coef=0.5,  
    max_grad_norm=0.5,  
    use_sde=False,  # Not needed for discrete action space
    target_kl=None,  
    tensorboard_log="./ppo_chess_logs/",
    verbose=1,  # Logs basic info
    device="auto",
    seed=42  # Ensures reproducibility
)


# Train agent for 1 million steps
model.learn(total_timesteps=1_000_000)

model.save('chess_agent_PPO_V0')

print("Model training complete and saved.")