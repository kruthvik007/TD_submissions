import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

# Neural network architecture for Q-learning
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, player=PLAYER1, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        # Assuming BOARD_SIZE x BOARD_SIZE input for the board state and output space for moves
        input_dim = BOARD_SIZE * BOARD_SIZE  # Flattened board
        output_dim = BOARD_SIZE * BOARD_SIZE * 2  # Potential actions

        # Initialize policy and target networks
        self.policy_net = DQNetwork(input_dim, output_dim)
        self.target_net = DQNetwork(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def choose_action(self, game):
        """Chooses an action based on epsilon-greedy policy."""
        possible_moves = self.get_possible_moves(game)
        
        if random.random() < self.epsilon:
            # Random choice for exploration
            return random.choice(possible_moves)
        else:
            # Exploitation: choose based on Q-values
            state = game.board.flatten()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            # Get the best action within the possible moves
            q_values_filtered = [(move, q_values[0][self.move_to_index(move)]) for move in possible_moves]
            best_move = max(q_values_filtered, key=lambda x: x[1])[0]
            return best_move

    def get_possible_moves(self, game):
        """Generates a list of valid moves formatted as in RandomAgent."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # Movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def move_to_index(self, move):
        """Encodes move as a unique index for Q-value lookup."""
        if len(move) == 2:  # Placement move
            return move[0] * BOARD_SIZE + move[1]
        elif len(move) == 4:  # Movement move
            return (move[0] * BOARD_SIZE + move[1]) * BOARD_SIZE * BOARD_SIZE + move[2] * BOARD_SIZE + move[3]

    def replay(self):
        """Trains the network with samples from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch elements to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor([self.move_to_index(action) for action in actions], dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q values for the current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values for next states
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Update policy network
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_best_move(self, game):
        """Interface for the agent to select the best move."""
        return self.choose_action(game)
