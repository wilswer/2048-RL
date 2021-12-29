import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    """Deep Q-learning neural network."""

    def __init__(self, lr, input_dims, n_actions):
        """Initialize the DeepQNetwork."""
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        #  self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        #  self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        #  self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)

        #  def conv2d_size_out(size, kernel_size=3, stride=2):
        #      return (size - (kernel_size - 1) - 1) // stride + 1
        #  h = self.input_dims[1]
        #  w = self.input_dims[2]
        #  convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #  convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #  linear_input_size = convw * convh * 32
        #  self.head = nn.Linear(linear_input_size, self.n_actions)
        self.fc1 = nn.Linear(*self.input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """Forward pass in the network."""
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    """Deep Q-learning agent."""

    def __init__(
            self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end=0.01, eps_decay=5e-6,
    ):
        """Initialize the agent."""
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, self.input_dims, n_actions)

        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """Store transition in memory."""
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.reward_memory[idx] = reward
        self.action_memory[idx] = action
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        """Choose an action with epsilon-greedy strategy."""
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        """Train the agent."""
        if self.mem_cntr < self.mem_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(
            self.Q_eval.device
        )
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(
            self.Q_eval.device
        )
        reward_batch = T.tensor(self.reward_memory[batch]).to(
            self.Q_eval.device
        )
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(
            self.Q_eval.device
        )
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_min)
