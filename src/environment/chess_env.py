import gymnasium as gym
import numpy as np
import chess

# board = chess.Board()

# print(board)

class ChessEnv(gym.Env):

    def __init__(self):
        super(ChessEnv, self).__init__()

        # Chess board & logic initialization
        self.board = chess.Board()

        # Box space used for the observation space
        self.observation_space = gym.spaces.Box(
            low=-6, # Black pieces
            high=6, # White pieces
            shape=(8,8), # Board size
            dtype=np.int8 # Discrete values in space
        )

        # discrete range to cover all possible moves in chess
        self.action_space = gym.spaces.Discrete(4672)

        # Maps chess pieces to upper & lower bound elements in observation space
        self.pieces = {
            # White pieces
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,

            # Black pieces
            'p': -1,
            'n': -2,
            'b': -3,
            'r': -4,
            'q': -5,
            'k': -6,
        }

        # Track whose turn it is:  True for White, False for Black
        self.current_player = True

        self.done = False # Determines if the enviornment has been terminated


    # Maps the current chess board state to the current observation state
    def get_observation(self):

        # Initialize an empty board
        observation = np.zeros((8,8), dtype=np.int8)

        # Map pieces to observation values
        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row = chess.square_rank(square)
            col = chess.square_file(square)
            observation[7 - row][col] = self.pieces[piece.symbol()]  # Flip row for display
        
        return observation







    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options) # Implements correct seeding
        self.board.reset() # restarts the game board
        self.current_player = True # White makes the first move
        observation = None

