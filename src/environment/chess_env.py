import gymnasium as gym
import numpy as np
import chess


class ChessEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(ChessEnv, self).__init__()

        # Chess board & logic initialization
        # Will be used to shape the observation space
        self.board = chess.Board()

        # Stores a list of all possible moves (4096) legal or illegal 
        self.all_possible_moves = self.generate_all_moves()

        # Box space used for the observation space
        self.observation_space = gym.spaces.Box(
            low=-6, # Black pieces
            high=6, # White pieces
            shape=(8,8), # Board size
            dtype=np.int8 # Discrete values in space
        )

        # discrete range to cover almost all possible moves in chess
        self.action_space = gym.spaces.Discrete(4672)

        # Maps chess pieces to upper & lower bound elements in observation space
        self.pieces = self.map_pieces()

        # Maps all chess move objects to an integer
        self.move_to_action = self.get_move_to_action()

        # Maps integers to all chess move objects
        self.action_to_move = self.get_action_to_move()

        # Track whose turn it is:  True for White, False for Black
        self.current_player = True

        self.done = False # Determines if the enviornment has been terminated

    


    # Maps chess pieces to upper & lower bound elements in observation space 
    def map_pieces(self):
        pieces = {
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

        return pieces


    # Generates a list of all possible moves legal or illeagal
    def generate_all_moves(self):
        
        moves = [] # initialize empty list

        # Iterates thru the entire board of squares 0-63 
        for from_square in chess.SQUARES: # Starting position
            for to_square in chess.SQUARES: # Ending position
                # Converts integer squares into a chess move
                move = chess.Move(from_square, to_square)
                moves.append(move) # adds move to the list
        
        return moves
    
    # Maps the chess move object to an integer
    def get_move_to_action(self):
        
        move_to_action = {} # Empty dictionary

        # action is the index (integer) which will be stored as the value
        # for move_to_action. move will be the key
        for action, move in enumerate(self.all_possible_moves):
            # if the key doesn't already exist
            if move not in move_to_action:
                move_to_action[move] = action # initialize key-value pair

        return move_to_action
    

    # Maps integers to chess move objects
    def get_action_to_move(self):

        action_to_move = {} # empty dictionary

        # iterates thru each key value pair in move_to_action dictionary
        # Where action (an integer) will be the key in action_to_move
        # and move (a chess move object) will be the value
        for move, action in self.move_to_action.items():
            if action not in action_to_move:
                action_to_move[action] = move

        return action_to_move



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
        observation = self.get_observation()

        return observation


    # Defines reward system for the learning model
    def compute_reward(self):
        pass

    # Decodes action sample into a chess move
    def decode_action(self, action):
        return self.action_to_move[action]


    def step(self, action):
        return super().step(action)




    # Defines rendering for human visualization
    def render(self, mode='human'):
        # Render the board for the user
        print(self.board)