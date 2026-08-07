import unittest
import os
import shutil

from src.agents.ppo_agent import create_model


class TestPPOAgent(unittest.TestCase):
    """Smoke‑test for the PPO model factory."""

    def test_create_model_loads_without_error(self):
        log_dir = "./tmp_test_ppo_logs/"
        model = None
        try:
            model = create_model(tensorboard_log=log_dir)
            self.assertIsNotNone(model)
        finally:
            if model is not None:
                model.env.close()
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()