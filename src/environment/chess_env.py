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

        # Tracks total number of illegal moves per episode
        self.illegal_moves_count = 0

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
        observation = self.get_observation() # returns game board as an 8x8 observation grid

        # stores additional metada about the initial observation state
        info = {
            "initial_board_fen": self.board.fen(),
            "starting_player": "white",
            "castling_rights": self.board.castling_rights,
            "en_passant_square": self.board.ep_square,
        }

        return observation, info



    # Decodes action sample into a chess move
    def decode_action(self, action):
        if action not in self.action_to_move:
            raise ValueError(f"Invalid action index: {action}")
        return self.action_to_move[action]
    

    
    # Computes reward for piece capture
    def reward_for_capture(self, move, previous_board):

        # checks if chess move resulted in a capture
        if self.board.is_capture(move):

            captured_piece = previous_board.piece_at(move.to_square)

            # Assign rewards based on the piece value
            piece_values = {'p': 0.1, 'n': 0.3, 'b': 0.3, 'r': 0.5, 'q': 0.9, 'k': 0}  # King capture not possible
            reward = piece_values[captured_piece.symbol().lower()]
            
            # Positive reward for capturing an opponent piece
            if captured_piece.color != self.current_player:
                return reward
        
        return 0 # if the move results in no capture of opponent piece



    # Reward the agent for placing the opponent in check.
    def reward_for_check(self):
        if self.board.is_check():
            return 0.5  # Positive reward for putting the opponent in check
        return 0



    # Calculates the current material balance advantage/disadvantage the agent has
    def calculate_material_balance(self, board):

        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

        agent_color = self.current_player # determines if the piece belongs to the agent

        agent_balance = 0 # agents piece material
        opponent_balance = 0 # opponent's material

        material_balance = 0 # Tracks material balance for the agent

        piece_map = board.piece_map() # Gathers all pieces on the board

        for square, piece in piece_map.items():

            # If piece belongs to the agent
            if piece.color == agent_color:  
                agent_balance += piece_values.get(piece.symbol().lower(), 0)
            # otherwise it is the opponents
            else:
                opponent_balance += piece_values.get(piece.symbol().lower(), 0)

        
        max_material = 39  # Sum of all pieces excluding kings

        material_balance = agent_balance - opponent_balance # returns the agent's material balance
        material_balance = material_balance / max_material # normalize balance

        return material_balance
    


    # Gives reward for pieces in the central squares
    def calculate_central_control(self, board):

        # total number of pieces in the central control area
        central_control = 0

        # Extended Central squares
        central_squares = [

            chess.C3, 
            chess.C4, 
            chess.C5, 
            chess.C6,

            chess.D3, 
            chess.D4, 
            chess.D5, 
            chess.D6,

            chess.E3,
            chess.E4, 
            chess.E5,
            chess.E6,

            chess.F3,
            chess.F4, 
            chess.F5,
            chess.F6,
        ]

        for square in central_squares:
            piece = board.piece_at(square)
            # reward for every agent piece in the central control area
            if piece and piece.color == self.current_player:
                central_control+=0.1
        
        return central_control


    
    def calculate_king_safety(self, board):

        # The square int (0-63) that the agent's king piece is on
        agent_king_square = board.king(self.current_player)

        king_safety_score = 0 # Return value

        is_pawn_in_file = False

        if agent_king_square is not None:

            king_file = chess.square_file(agent_king_square) 

            # Is true if there is any pawn piece in the same file (column) as the king
            for square, piece in board.piece_map().items():

                current_file = chess.square_file(square)             

                if current_file == king_file:
                    # if the piece in the same file is a pawn & belongs to the agent
                    if piece.piece_type == chess.PAWN and piece.color == self.current_player:
                        is_pawn_in_file = True
            
            # penalty for king being exposed
            if is_pawn_in_file == False:
                king_safety_score-= 0.5


            king_rank = chess.square_rank(agent_king_square)

            # Check the file (column) to the left, same file, and right in 
            # the rank (row) directly above the king
            for offset in [-1, 0, 1]: 
                
                square = None  

                if self.current_player: # if the agent is currently White
                    square = chess.square(king_file + offset, king_rank + 1)

                else: # if the agent is currently Black
                    square = chess.square(king_file + offset, king_rank - 1)
                
                # Stores the piece at the current square
                # returns None if there is no piece present
                piece = board.piece_at(square)

                # checks if the current piece is a pawn
                if piece and piece.piece_type == chess.PAWN:
                    king_safety_score += 0.2


        return king_safety_score




    # Gives total reword for positioning and material balance
    def reward_for_position(self, previous_board):

        # Material balance before & after step
        previous_material = self.calculate_material_balance(previous_board)
        current_material = self.calculate_material_balance(self.board)
        material_balance = current_material - previous_material

        # total number of agent pieces in the central control area
        previous_central_control = self.calculate_central_control(previous_board)
        current_central_control = self.calculate_central_control(self.board)
        positional_reward = current_central_control - previous_central_control

        # King safety
        previous_king_safety = self.calculate_king_safety(previous_board)
        current_king_safety = self.calculate_king_safety(self.board)
        king_safety_reward = current_king_safety - previous_king_safety

        total_reward = material_balance + positional_reward + king_safety_reward


        return total_reward





    # Defines reward system for the learning model
    def compute_reward(self, move, previous_board):

        total_reward = 0 # total reward value returned
        
        # Calculate each reward parameter
        capture_reward = self.reward_for_capture(move, previous_board)
        check_reward = self.reward_for_check()
        position_reward = self.reward_for_position(previous_board)

        # if either players king is in checkmate
        if self.board.is_checkmate(): 
            # If the agent is not in checkmate, it will 
            # receive a positive reward for winning the game
            if self.board.turn != self.current_player:
                total_reward+=1
            else: # if the agent loses the game
                total_reward+=-1
            
        # If the game ends in a draw
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            total_reward+=0.5

        total_reward = total_reward + capture_reward + check_reward + position_reward

        return total_reward




    # Logic for when agent takes a single action step
    def step(self, action):
        
        # if the agent picks an illegal move in a given postion
        illegal_move_penalty = -0.1 

        legal_move_made = False # exit condition for while loop

        if action not in self.action_to_move:
            raise ValueError(f"Invalid action index: {action}")
    
         # Converts int action into a chess move object
        move = self.action_to_move[action] 
        previous_board = self.board.copy()  # Save the board state before the move


        # Check for stalemates or insufficient material before decoding the action
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            
            observation = self.get_observation()
            reward = 0.5  # Draw reward
            done = True
            info = {
                "reason": "draw",
                "draw_reason": "stalemate" if self.board.is_stalemate() else "insufficient_material",
                "illegal_moves": self.illegal_moves_count,
                "total_moves_made": self.board.fullmove_number,
                "in_check": self.board.is_check()
            }
            return observation, reward, done, False, info
        

        # Ensures longer games allow slightly more mistakes, while shorter games are stricter.
        threshold = max(5, self.board.fullmove_number // 2)

        if self.illegal_moves_count > threshold:
            observation = self.get_observation()
            reward = -1  # Penalize for too many illegal moves
            done = True
            info = {
                "reason": "too many illegal moves",
                "illegal_moves": self.illegal_moves_count,
                "total_moves_made": self.board.fullmove_number,
                "in_check": self.board.is_check()
            }

            print(f"Threshold for illegal moves: {threshold}")

            return observation, reward, done, False, info


        while not legal_move_made:

            # Checks if move is legal
            if move in self.board.legal_moves:
                self.board.push(move) # applies move to the board object
                self.current_player = not self.current_player # Changes player turn
                legal_move_made = True # breaks while loop
            else:
                observation = self.get_observation()
                done = False # indicates that the game is not over
                info = {
                    "reason": "illegal move",
                    "illegal_moves": self.illegal_moves_count,
                    "total_moves_made": self.board.fullmove_number,
                    "in_check": self.board.is_check()
                }
                
                self.illegal_moves_count+=1 # Tracks total number of illegal moves per episode
                

                return observation, illegal_move_penalty, done, False, info
                
        
        reward = self.compute_reward(move, previous_board) # Compute total reward for the step

        done = self.board.is_game_over() # Check if the game is over

        observation = self.get_observation() # Get the updated observation

        # additional metadata info for debugging & analysis
        info = {
            "illegal_moves": self.illegal_moves_count,
            "total_moves_made": self.board.fullmove_number,
            "in_check": self.board.is_check()
        }

        # if the game ended
        if done:
            if self.board.result() == "1-0":
                info["winner"] = "white"
            elif self.board.result() == "0-1":
                info["winner"] = "black"
            else:
                info["winner"] = "draw"

        
        return observation, reward, done, False, info




    # Render the sequence of moves made so far in the game.
    def render_move_history(self):
        
        move_history = list(self.board.move_stack)

        print("Move History:")

        for i, move in enumerate(move_history):
            print(f"{i+1}. {move}")



    # Display all legal moves from the current position
    def render_legal_moves(self):

        legal_moves = list(self.board.legal_moves)
        print("Legal Moves:")
        print([move.uci() for move in legal_moves])


    # Defines rendering for human visualization
    def render(self, mode='human'):
        # Render the board for the user
        print(self.board.unicode(invert_color=True))

        # Render additional information
        print("\n=== Additional Info ===\n")
        self.render_move_history()
        print("\n")
        self.render_legal_moves()
        print("\n")