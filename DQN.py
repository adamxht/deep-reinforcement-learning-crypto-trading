# Class implementation of Duelling Deep Q Network

import torch
from random import sample, random, choice
from collections import namedtuple, deque
import torch.nn as nn
from torch import optim
import numpy as np


STATE_SPACE = 132
ACTION_SPACE = 3

GAMMA = 0.9995
TAU = 1e-3

MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200

LR_DQN = 5e-5

LEARN_AFTER = MEMORY_THRESH
LEARN_EVERY = 3
UPDATE_EVERY = 9

gpu_available = torch.cuda.is_available()
if gpu_available:
    DEVICE = "cuda"
else :
    DEVICE = "cpu"

class ReplayMemory:

    def __init__(self, capacity=MEMORY_LEN):
        self.memory = deque(maxlen=capacity)

    def store(self, t):
        self.memory.append(t)

    def sample(self, n):
        a = sample(self.memory, n)
        return a

    def __len__(self):
        return len(self.memory)
    
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1]
        
    
class DuellingDQN(nn.Module):
  """
  Acrchitecture for Duelling Deep Q Network Agent
  """

  def __init__(self, input_dim=STATE_SPACE, output_dim=ACTION_SPACE, hidden_size = 100):
    super(DuellingDQN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.fc1 = nn.Linear(self.input_dim, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, 300)
    self.fc4 = nn.Linear(300, 200)
    self.fc5 = nn.Linear(200, 10)

    self.fcs = nn.Linear(10, 1)
    self.fcp = nn.Linear(10, self.output_dim)
    self.fco = nn.Linear(self.output_dim + 1, self.output_dim)

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.sig = nn.Sigmoid()
    self.sm = nn.Softmax(dim=1)

# Forward pass
  def forward(self, state):
    x = self.relu(self.fc1(state))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.relu(self.fc4(x))
    x = self.relu(self.fc5(x))
    xs = self.relu(self.fcs(x))
    xp = self.relu(self.fcp(x))

    x = xs + xp - xp.mean()
    return x

class DQNAgent:
  """
  Implements the Agent components
  """

  def __init__(self, actor_net=DuellingDQN, memory=ReplayMemory()):
    self.actor_online = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
    self.actor_target = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
    self.actor_target.load_state_dict(self.actor_online.state_dict())
    self.actor_target.eval()

    self.memory = memory

    self.actor_criterion = nn.MSELoss()
    self.actor_op = optim.Adam(self.actor_online.parameters(), lr=LR_DQN)

    self.t_step = 0

# Action
  def act(self, state, eps=0.):
    self.t_step += 1
    state = torch.from_numpy(state).float().to(DEVICE).view(1, -1)
    
    self.actor_online.eval()
    with torch.no_grad():
        actions = self.actor_online(state)
    self.actor_online.train()

    if random() > eps:
        act = np.argmax(actions.cpu().data.numpy())
    else:
        act = choice(np.arange(ACTION_SPACE))
    return int(act)


  def learn(self):
    if len(self.memory) <= MEMORY_THRESH:
        return 0

    if self.t_step > LEARN_AFTER and self.t_step % LEARN_EVERY==0:
        
    # Sample experiences from the Memory
        batch = self.memory.sample(BATCH_SIZE)

        states = np.vstack([t.States for t in batch])
        states = torch.from_numpy(states).float().to(DEVICE)

        actions = np.vstack([t.Actions for t in batch])
        actions = torch.from_numpy(actions).float().to(DEVICE)

        rewards = np.vstack([t.Rewards for t in batch])
        rewards = torch.from_numpy(rewards).float().to(DEVICE)

        next_states = np.vstack([t.NextStates for t in batch])
        next_states = torch.from_numpy(next_states).float().to(DEVICE)

        dones = np.vstack([t.Dones for t in batch]).astype(np.uint8)
        dones = torch.from_numpy(dones).float().to(DEVICE)

        # ACTOR UPDATE
        # Compute next state actions and state values
        next_state_values = self.actor_target(next_states).max(1)[0].unsqueeze(1)
        y = rewards + (1-dones) * GAMMA * next_state_values
        state_values = self.actor_online(states).gather(1, actions.type(torch.int64))
        # Compute Actor loss
        actor_loss = self.actor_criterion(y, state_values)
        # Minimize Actor loss
        self.actor_op.zero_grad()
        actor_loss.backward()
        self.actor_op.step()

        if self.t_step % UPDATE_EVERY == 0:
            self.soft_update(self.actor_online, self.actor_target)
            
  # Soft updates improves the stability of learning
  def soft_update(self, local_model, target_model, tau=TAU):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

