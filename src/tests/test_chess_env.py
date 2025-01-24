import unittest
import numpy as np
import chess
from environment.chess_env import ChessEnv

class TestChessEnv(unittest.TestCase):

    def setUp(self):
        # Initialize the environment
        self.env = ChessEnv()


    ###########################################
    ###### get_observation unit testing #######
    ###########################################

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








    ###########################################
    ###### render unit testing ################
    ###########################################

    def test_render(self):
        board = chess.Board()
        self.assertEqual(self.env.board, board, "Should be rendered correctly")







    ##########################################
    ###### reset unit testing ################
    ##########################################

    def test_reset(self):
        # Starting position board
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

    
        self.env.board.push_uci('e2e4') # change board status
        observation = self.env.reset() # reset board to starting position

        np.testing.assert_array_equal(observation, expected_observation, "Should reset board back to starting position")

        self.env.reset() # reset the environment again

        # Check that the turn is set to the initial player (e.g., White)
        self.assertTrue(
            self.env.current_player, "Reset did not set the initial player to White."
        )

        # Ensure the game is not done after a reset
        self.assertFalse(
            self.env.done, "Reset did not properly set the game state to not done."
        )


    def tearDown(self):
        # Clean up resources if needed
        self.env.close()

if __name__ == '__main__':
    unittest.main()