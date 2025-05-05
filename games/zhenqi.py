import datetime
import math
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = 1

        self.observation_shape = (3, 8, 8)
        self.action_space = list(range(8 * 8))
        self.players = list(range(2))
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = "random"

        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 100
        self.num_simulations = 200 #后期改成400
        self.discount = 1
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 2000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 500

        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 100  # Number of game moves to keep for every batch element
        self.td_steps = 100  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = Zhenqi()

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)


class Zhenqi:
    def __init__(self):
        self.board_size = 8
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]
        self.history = 0
    # 0 为先手 1 为后手
    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        self.history = 0
        return self.get_observation()

    def step(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size

        assert self.board[x][y] == 0, f"Invalid action x:{x}, y:{y}"
        self.board[x][y] = self.player
        self.history += 1

        done = self.is_finished()
        reward = 1 if done == self.player else -1 if done == self.player*-1 else 0


        if done != 0:
            self.player *= -1
            return self.get_observation(), reward, True if done != 0 else False

        # 震开四周棋子
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if x + dx >= 0 and x + dx < self.board_size and y + dy >= 0 and y + dy < self.board_size:
                    self.move_piece((x + dx, y + dy), (x + dx + dx, y + dy + dy))

        done = self.is_finished()
        his_reward = 0
        if self.history > 10 and self.history < 30:
            his_reward = -0.02
        elif self.history >= 30 and self.history < 50:
            his_reward = -0.04
        elif self.history >= 50 and self.history < 80:
            his_reward = -0.07
        elif self.history >= 80:
            his_reward = -0.09
        reward = 1 if done == self.player else -1 if done == self.player*-1 else his_reward
        self.player *= -1

        return self.get_observation(), reward, True if done != 0 else False

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((self.board_size, self.board_size), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                color = self.board[x][y]
                if color == 0:
                    continue
                if x+3 < self.board_size and self.board[x+1][y]==color and self.board[x+2][y]==color and self.board[x+3][y]==color:
                    return color
                if y+3 < self.board_size and self.board[x][y+1]==color and self.board[x][y+2]==color and self.board[x][y+3]==color:
                    return color
                if x+3 < self.board_size and y+3 < self.board_size and self.board[x+1][y+1]==color and self.board[x+2][y+2]==color and self.board[x+3][y+3]==color:
                    return color
                if x-3 >= 0 and y+3 < self.board_size and self.board[x-1][y+1]==color and self.board[x-2][y+2]==color and self.board[x-3][y+3]==color:
                    return color
        return 0

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y

    def move_piece(self, move1, move2):
        "震棋子move 1 到 move 2"
        (x1, y1) = move1
        (x2, y2) = move2

        if x2 > self.board_size - 1 or x2 < 0 or y2 > self.board_size - 1 or y2 < 0:
            self.board[x1][y1] = 0
            return

        if self.board[x2][y2] == 0 and self.board[x1][y1] != 0:
            self.board[x2][y2] = self.board[x1][y1]
            self.board[x1][y1] = 0
