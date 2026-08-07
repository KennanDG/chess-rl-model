"""Simplified, curriculum based chess training environment.

The full :class:`environment.chess_env.ChessEnv` asks the agent to learn a whole
game of chess at once, which makes PPO training unstable.  This module breaks
the problem into small reproducible stages that still obey the movement rules of
real chess pieces (all legal moves come from ``python-chess``), so later stages
can be grown towards a full game without changing the agent interface.

Curriculum stages, in order:
    1. ``reach_square``  - move a single piece to a target square.
    2. ``capture_piece`` - capture a single static opponent piece.

Both stages keep the standard Gymnasium ``reset``/``step`` contract used by the
full environment, and :meth:`SimpleChessEnv.legal_actions` is exposed so action
masking can be added when the curriculum is extended.
"""

import chess
import gymnasium as gym
import numpy as np

# Stage identifiers
STAGE_REACH_SQUARE = "reach_square"
STAGE_CAPTURE_PIECE = "capture_piece"

# Ordered curriculum: each entry is a reproducible step towards full chess.
STAGES = (STAGE_REACH_SQUARE, STAGE_CAPTURE_PIECE)

# Pawns are excluded as the agent piece for now because promotions need extra
# action encoding.  They become available once promotions are added.
AGENT_PIECE_TYPES = {
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

OPPONENT_PIECE_TYPES = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
}

# Same board encoding used by the full chess environment.
PIECE_ENCODING = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
}

# Bonus on top of the goal reward, matching the capture values of the full env.
CAPTURE_PIECE_VALUES = {
    chess.PAWN: 0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.3,
    chess.ROOK: 0.5,
    chess.QUEEN: 0.9,
}

BOARD_SQUARES = 64


class SimpleChessEnv(gym.Env):
    """Single piece chess task with a dense, easy to debug reward signal."""

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self,
        stage=STAGE_REACH_SQUARE,
        agent_piece="knight",
        opponent_piece="pawn",
        max_steps=32,
        max_illegal_moves=8,
        step_penalty=-0.01,
        illegal_move_penalty=-0.1,
        goal_reward=1.0,
        distance_reward_scale=0.1,
    ):
        super(SimpleChessEnv, self).__init__()

        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage!r}. Expected one of {STAGES}.")
        if agent_piece not in AGENT_PIECE_TYPES:
            raise ValueError(
                f"Unknown agent piece: {agent_piece!r}. "
                f"Expected one of {sorted(AGENT_PIECE_TYPES)}."
            )
        if opponent_piece not in OPPONENT_PIECE_TYPES:
            raise ValueError(
                f"Unknown opponent piece: {opponent_piece!r}. "
                f"Expected one of {sorted(OPPONENT_PIECE_TYPES)}."
            )

        self.stage = stage
        self.agent_piece = agent_piece
        self.opponent_piece = opponent_piece
        self.agent_piece_type = AGENT_PIECE_TYPES[agent_piece]
        self.opponent_piece_type = OPPONENT_PIECE_TYPES[opponent_piece]

        self.max_steps = max_steps
        self.max_illegal_moves = max_illegal_moves
        self.step_penalty = step_penalty
        self.illegal_move_penalty = illegal_move_penalty
        self.goal_reward = goal_reward
        self.distance_reward_scale = distance_reward_scale

        # Plane 0: piece encoding, plane 1: goal square marker.
        self.observation_space = gym.spaces.Box(
            low=-6,
            high=6,
            shape=(2, 8, 8),
            dtype=np.int8,
        )

        # Action index = from_square * 64 + to_square.
        self.action_space = gym.spaces.Discrete(BOARD_SQUARES * BOARD_SQUARES)

        self.agent_color = chess.WHITE
        self.board = chess.Board(None)  # Empty board until reset()
        self.agent_square = None
        self.target_square = None
        self.steps = 0
        self.illegal_moves_count = 0
        self.done = False

    ###################################################
    ###### Action encoding / decoding helpers #########
    ###################################################

    def encode_action(self, move):
        """Maps a chess move object to its discrete action index."""
        return move.from_square * BOARD_SQUARES + move.to_square

    def decode_action(self, action):
        """Maps a discrete action index to a chess move object."""
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"Invalid action index: {action}")
        return chess.Move(action // BOARD_SQUARES, action % BOARD_SQUARES)

    def legal_actions(self):
        """Action indices of every legal move, useful for future action masking."""
        return [self.encode_action(move) for move in self.board.legal_moves]

    ###################################################
    ###### Board / observation helpers ################
    ###################################################

    def get_observation(self):
        observation = np.zeros((2, 8, 8), dtype=np.int8)

        for square, piece in self.board.piece_map().items():
            row = 7 - chess.square_rank(square)  # Flip row for display
            col = chess.square_file(square)
            observation[0][row][col] = PIECE_ENCODING[piece.symbol()]

        if self.target_square is not None:
            row = 7 - chess.square_rank(self.target_square)
            col = chess.square_file(self.target_square)
            observation[1][row][col] = 1

        return observation

    def distance_to_target(self):
        """Chebyshev distance between the agent piece and the goal square."""
        if self.agent_square is None or self.target_square is None:
            return -1
        return chess.square_distance(self.agent_square, self.target_square)

    def _validate_square(self, square, name):
        square = int(square)
        if not 0 <= square < BOARD_SQUARES:
            raise ValueError(f"{name} must be in [0, 63], got {square}")
        return square

    def _is_valid_target(self, agent_square, target_square):
        # A bishop can only ever reach squares of its own colour.
        if self.agent_piece_type == chess.BISHOP:
            return self._square_color(agent_square) == self._square_color(target_square)
        return True

    @staticmethod
    def _square_color(square):
        return (chess.square_file(square) + chess.square_rank(square)) % 2

    def _sample_target_square(self, agent_square):
        candidates = [
            square for square in chess.SQUARES
            if square != agent_square and self._is_valid_target(agent_square, square)
        ]
        return int(self.np_random.choice(candidates))

    def _build_board(self):
        self.board = chess.Board(None)  # Empty board, no castling rights
        self.board.turn = self.agent_color
        self.board.set_piece_at(
            self.agent_square,
            chess.Piece(self.agent_piece_type, self.agent_color),
        )

        if self.stage == STAGE_CAPTURE_PIECE:
            self.board.set_piece_at(
                self.target_square,
                chess.Piece(self.opponent_piece_type, not self.agent_color),
            )

    def _build_info(self, reason):
        return {
            "stage": self.stage,
            "reason": reason,
            "steps": self.steps,
            "illegal_moves": self.illegal_moves_count,
            "agent_square": chess.square_name(self.agent_square),
            "target_square": chess.square_name(self.target_square),
            "distance_to_target": self.distance_to_target(),
            "legal_moves": len(list(self.board.legal_moves)),
        }

    ###################################################
    ###### Gymnasium API ##############################
    ###################################################

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed, options=options)  # Implements correct seeding

        options = options or {}

        agent_square = options.get("agent_square")
        if agent_square is None:
            agent_square = int(self.np_random.integers(BOARD_SQUARES))
        else:
            agent_square = self._validate_square(agent_square, "agent_square")

        target_square = options.get("target_square")
        if target_square is None:
            target_square = self._sample_target_square(agent_square)
        else:
            target_square = self._validate_square(target_square, "target_square")

        if agent_square == target_square:
            raise ValueError("agent_square and target_square must be different")

        self.agent_square = agent_square
        self.target_square = target_square
        self.steps = 0
        self.illegal_moves_count = 0
        self.done = False

        self._build_board()

        observation = self.get_observation()
        info = self._build_info("reset")
        info["initial_board_fen"] = self.board.fen()

        return observation, info

    def step(self, action):

        move = self.decode_action(action)
        self.steps += 1

        # Illegal moves are penalised instead of applied, which keeps the stage
        # aligned with the movement rules of the chosen piece.
        if move not in self.board.legal_moves:
            self.illegal_moves_count += 1
            terminated = self.illegal_moves_count >= self.max_illegal_moves
            truncated = not terminated and self.steps >= self.max_steps
            self.done = terminated or truncated
            reason = "too_many_illegal_moves" if terminated else "illegal_move"
            return (
                self.get_observation(),
                self.illegal_move_penalty,
                terminated,
                truncated,
                self._build_info(reason),
            )

        previous_distance = self.distance_to_target()
        captured_piece = self.board.piece_at(move.to_square)

        self.board.push(move)

        # These stages only train the agent's own piece, so the agent keeps the
        # move.  Progressing towards full chess means dropping this line and
        # letting the opponent reply instead.
        self.board.turn = self.agent_color
        self.agent_square = move.to_square

        reward = self.step_penalty
        goal_reached = False

        if self.stage == STAGE_REACH_SQUARE:
            if self.agent_square == self.target_square:
                reward += self.goal_reward
                goal_reached = True
        elif captured_piece is not None and captured_piece.color != self.agent_color:
            reward += self.goal_reward
            reward += CAPTURE_PIECE_VALUES.get(captured_piece.piece_type, 0.0)
            goal_reached = True

        # Dense shaping so the agent learns to move towards the goal.
        if not goal_reached:
            reward += self.distance_reward_scale * (
                previous_distance - self.distance_to_target()
            )

        terminated = goal_reached
        truncated = not terminated and self.steps >= self.max_steps
        self.done = terminated or truncated

        if terminated:
            reason = "goal_reached"
        elif truncated:
            reason = "max_steps"
        else:
            reason = "in_progress"

        return (
            self.get_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(reason),
        )

    def render(self, mode='human'):
        print(
            f"Stage: {self.stage} | step {self.steps}/{self.max_steps} | "
            f"illegal moves: {self.illegal_moves_count}"
        )
        if self.target_square is not None:
            print(f"Target square: {chess.square_name(self.target_square)}")

        print(self.board.unicode(invert_color=True))
        print(f"Legal moves: {[move.uci() for move in self.board.legal_moves]}\n")
