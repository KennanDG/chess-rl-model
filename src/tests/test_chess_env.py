import unittest
import numpy as np
import chess
from environment.chess_env import ChessEnv

class TestChessEnv(unittest.TestCase):

    def setUp(self):
        # Initialize the environment
        self.env = ChessEnv()



    ###########################################
    ###### map_pieces unit testing ############
    ###########################################
    
    def test_map_pieces(self):
        pieces = self.env.map_pieces()
        expected_pieces = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        self.assertEqual(pieces, expected_pieces, "Piece mapping is incorrect.")



    ##############################################
    ###### generate_all_moves unit testing #######
    ##############################################

    def test_generate_all_moves(self):
        all_moves = self.env.generate_all_moves()
        self.assertEqual(len(all_moves), 4096, "All possible moves list should contain 4096 moves.")
        self.assertTrue(all(isinstance(move, chess.Move) for move in all_moves), "All moves should be chess.Move objects.")



    ###########################################################
    ###### move_to_action & action_to_move unit testing #######
    ###########################################################

    def test_get_move_to_action(self):
        move_to_action = self.env.get_move_to_action()
        self.assertEqual(len(move_to_action), 4096, "Move to action mapping should have 4096 entries.")
        self.assertIsInstance(move_to_action[self.env.all_possible_moves[0]], int, "Move to action values should be integers.")

    def test_get_action_to_move(self):
        action_to_move = self.env.get_action_to_move()
        self.assertEqual(len(action_to_move), 4096, "Action to move mapping should have 4096 entries.")
        self.assertIsInstance(action_to_move[0], chess.Move, "Action to move values should be chess.Move objects.")



    ###########################################
    ###### get_observation unit testing #######
    ###########################################

    def test_get_observation(self):
        observation = self.env.get_observation()
        self.assertEqual(observation.shape, (8, 8), "Observation should be an 8x8 array.")
        self.assertTrue(np.all(np.isin(observation, range(-6, 7))), "Observation values should be in the range [-6, 6].")



    ##########################################
    ###### reset unit testing ################
    ##########################################

    def test_reset(self):
        observation, info = self.env.reset()
        self.assertEqual(observation.shape, (8, 8), "Reset observation should be an 8x8 array.")
        self.assertIn("initial_board_fen", info, "Reset info should include 'initial_board_fen'.")
        self.assertEqual(info["starting_player"], "white", "Starting player should be white.")



    ###########################################
    ###### render unit testing ################
    ###########################################

    def test_render(self):
        board = chess.Board()
        self.assertEqual(self.env.board, board, "Should be rendered correctly")



    ###########################################
    ###### decode_action unit testing #########
    ###########################################

    def test_decode_action(self):
        move = self.env.decode_action(0)
        self.assertIsInstance(move, chess.Move, "Decoded action should be a chess.Move object.")
        with self.assertRaises(ValueError):
            self.env.decode_action(9999)  # Invalid action index



    ################################################
    ###### reward_for_capture unit testing #########
    ################################################

    def test_reward_for_capture(self):
        self.env.board.set_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        previous_board = self.env.board.copy()
        move = chess.Move.from_uci("e4d5")  # White captures pawn on d5
        self.env.board.push(move)
        reward = self.env.reward_for_capture(move, previous_board)
        self.assertEqual(reward, 0.1, "Reward for capturing a pawn should be 0.1.")



    ################################################
    ###### reward_for_check unit testing ###########
    ################################################

    def test_reward_for_check(self):
        self.env.board.set_fen("rnbqkbnr/pppp1ppp/8/8/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 5")
        reward = self.env.reward_for_check()
        self.assertEqual(reward, 0.5, "Reward for placing opponent in check should be 0.5.")



    ##########################################################
    ###### calculate_material_balance unit testing ###########
    ##########################################################

    def test_calculate_material_balance(self):
        # Initializes board where white is up one pawn
        self.env.board.set_fen('rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2')
        balance = self.env.calculate_material_balance(self.env.board)
        expected_balance = (1 / 39) 
        self.assertAlmostEqual(balance, expected_balance, delta=0.001, msg="Material balance is incorrect.")



    ##########################################################
    ###### calculate_central_control unit testing ############
    ##########################################################

    def test_calculate_central_control(self):
        self.env.board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        control = self.env.calculate_central_control(self.env.board)
        self.assertEqual(control, 0.1, "Central control should reward pieces in the center.")



    ##########################################################
    ###### calculate_king_safety unit testing ################
    ##########################################################

    def test_calculate_king_safety(self):
        self.env.board.set_fen("8/8/8/8/4K3/8/8/8 w - - 0 1") 
        safety = self.env.calculate_king_safety(self.env.board)
        self.assertEqual(safety, -0.5, "King safety should penalize exposed kings.")



    #################################################
    ###### illegal move unit testing ################
    #################################################

    def test_step_illegal_move(self):
        observation, info = self.env.reset()
        illegal_action = 9999  # Out of bounds action
        with self.assertRaises(ValueError):
            self.env.step(illegal_action)



    def tearDown(self):
        # Clean up resources if needed
        self.env.close()

if __name__ == '__main__':
    unittest.main()