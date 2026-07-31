from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.chess_env import ChessEnv

# Initialize Chess Evironment
env = DummyVecEnv([lambda: ChessEnv()])

# Balanced PPO hyperparameters for stable and efficient chess training
model = PPO(
    policy="MlpPolicy",  
    env=env,
    learning_rate=0.0003,  # Slightly increased for faster convergence
    n_steps=4096,  # Moderate rollout buffer for frequent updates
    batch_size=256,  # Standard batch size for stable gradient estimates
    n_epochs=10,  # Default number of epochs per update
    gamma=0.99,  
    gae_lambda=0.95,  
    clip_range=0.2,  # Standard clip range
    ent_coef=0.01,  # Moderate entropy to encourage exploration without excessive randomness
    vf_coef=0.5,  # Default value function coefficient for balanced policy/value trade-off
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