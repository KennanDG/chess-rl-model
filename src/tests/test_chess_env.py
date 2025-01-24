import unittest
import numpy as np
import chess
from environment.chess_env import ChessEnv

class TestChessEnv(unittest.TestCase):

    def setUp(self):
        # Initialize the environment
        self.env = ChessEnv()

    def test_get_observation_reset(self):
        # Reset the environment to the starting position
        self.env.board.reset()
        
        # Get the observation
        observation = self.env.get_observation()

        # Expected observation for the starting position
        expected_observation = np.array([
            [-4, -2, -3, -5, -6, -3, -2, -4],  # Black pieces
            [-1, -1, -1, -1, -1, -1, -1, -1],  # Black pawns
            [ 0,  0,  0,  0,  0,  0,  0,  0],  # Empty rows
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1],  # White pawns
            [ 4,  2,  3,  5,  6,  3,  2,  4],  # White pieces
        ], dtype=np.int8)

        # Validate the shape
        self.assertEqual(observation.shape, (8, 8), "Observation should be an 8x8 array.")

        # Validate the content
        np.testing.assert_array_equal(observation, expected_observation, "Initial board state mismatch.")

    def tearDown(self):
        # Clean up resources if needed
        self.env.close()

if __name__ == '__main__':
    unittest.main()