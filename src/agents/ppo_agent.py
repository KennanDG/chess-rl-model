"""PPO training entry point for the simplified chess curriculum.

Training starts on :data:`STAGE_REACH_SQUARE` (move one piece to a target
square) and can be repeated on :data:`STAGE_CAPTURE_PIECE` (capture a static
opponent piece), which keeps each training run small and reproducible.

Each stage is trained on its own environment instance so runs stay short,
seeded and independently reproducible. Stages can be chained with
:func:`train_curriculum`, which reuses the policy from the previous stage.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:  # Supports running with `src` on sys.path or importing `src.agents...`
    from environment.simple_chess_env import (
        SimpleChessEnv,
        STAGE_CAPTURE_PIECE,
        STAGE_REACH_SQUARE,
    )
except ImportError:  # pragma: no cover - import path fallback
    from src.environment.simple_chess_env import (
        SimpleChessEnv,
        STAGE_CAPTURE_PIECE,
        STAGE_REACH_SQUARE,
    )

DEFAULT_STAGE = STAGE_REACH_SQUARE
DEFAULT_AGENT_PIECE = "knight"
DEFAULT_TENSORBOARD_LOG = "./ppo_simple_chess_logs/"
DEFAULT_TIMESTEPS_PER_STAGE = 50_000

# Reproducible ordering of the curriculum: learn to travel the board legally
# first, then learn to capture.
CURRICULUM = (STAGE_REACH_SQUARE, STAGE_CAPTURE_PIECE)


def make_env(stage=DEFAULT_STAGE, agent_piece=DEFAULT_AGENT_PIECE):
    """Vectorised simplified chess environment for a single curriculum stage."""
    return DummyVecEnv([lambda: SimpleChessEnv(stage=stage, agent_piece=agent_piece)])


def create_model(
    env=None,
    stage=DEFAULT_STAGE,
    agent_piece=DEFAULT_AGENT_PIECE,
    tensorboard_log=DEFAULT_TENSORBOARD_LOG,
    seed=42,
):
    """PPO hyperparameters tuned for short, stable runs on the simple stages."""
    if env is None:
        env = make_env(stage=stage, agent_piece=agent_piece)

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=512,  # Short rollouts: episodes are only a few dozen steps
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Small bonus keeps exploration alive on sparse rewards
        vf_coef=0.5,
        max_grad_norm=0.5,  # Gradient clipping guards against unstable updates
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def train(
    stage=DEFAULT_STAGE,
    agent_piece=DEFAULT_AGENT_PIECE,
    total_timesteps=DEFAULT_TIMESTEPS_PER_STAGE,
    model=None,
    tensorboard_log=DEFAULT_TENSORBOARD_LOG,
    seed=42,
    save_path=None,
):
    """Train a single curriculum stage.

    Pass an existing ``model`` to continue training a policy learned on an
    earlier stage. Returns the trained model.
    """
    env = make_env(stage=stage, agent_piece=agent_piece)

    created_model = model is None
    if created_model:
        model = create_model(
            env=env,
            tensorboard_log=tensorboard_log,
            seed=seed,
        )
    else:
        model.set_env(env)

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=created_model,
        tb_log_name=f"ppo_{stage}",
    )

    if save_path is not None:
        model.save(save_path)

    return model


def train_curriculum(
    stages=CURRICULUM,
    agent_piece=DEFAULT_AGENT_PIECE,
    timesteps_per_stage=DEFAULT_TIMESTEPS_PER_STAGE,
    tensorboard_log=DEFAULT_TENSORBOARD_LOG,
    seed=42,
    save_prefix="simple_chess_ppo",
):
    """Run every stage in order, reusing the policy from the previous stage."""
    model = None

    for stage in stages:
        model = train(
            stage=stage,
            agent_piece=agent_piece,
            total_timesteps=timesteps_per_stage,
            model=model,
            tensorboard_log=tensorboard_log,
            seed=seed,
            save_path=f"{save_prefix}_{stage}",
        )

    return model


if __name__ == "__main__":
    train_curriculum()
