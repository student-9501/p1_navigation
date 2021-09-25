"""
implements a DQN agent to play the banana game
besides stock DQN, it implements soft updates and double DQN

agents can be run as standalone data gathering instances which do
no learning
"""
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim


# default parameters that can be overridden in the agent's constructor
BUFFER_SIZE = 100000
BATCH_SIZE = 64
TAU = 1e-3
LR = 5e-4
HIDDEN_SIZE = 512
GAMMA = 0.99  # discount factor
LEARN_EVERY = 4


class Model(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        
        self.seq = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_size),
        )

    def forward(self, x):
        """
        :param x: a float tensor of shape [state_size]
        :returns: a float tensor representing action values of shape [action_size]
        """
        return self.seq(x)


@dataclass
class Sample:
    """ a single sample in the replay buffer, which can be serialized
    and sent back from a data gathering agent to the learning process """
    state: list = None
    action: int = None
    reward: int = None
    next_state: list = None
    done: bool = None


class History:
    """ a list of samples that we can randomly sample for learning """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)

    def add_sample(self, sample: Sample):
        """ adds a sample, while discarding old entries

        :param sample: a single Sample to be added to the buffer
        """
        self.buffer.append(sample)

    def random_sample(self, batch_size: int):
        """
        :param batch_size: how many elements to return from the buffer
        :returns: a list of Sample
        """
        samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_size: int = 37, action_size: int = 4, model=None,
                 seed: int = 0, double_dqn: bool = False, batch_size: int = BATCH_SIZE,
                 buffer_size: int = BUFFER_SIZE, hidden_size: int = HIDDEN_SIZE,
                 lr: float = LR, gamma: float = GAMMA, tau: float = TAU,
                 learn_every=LEARN_EVERY, device_name: str = None):
        """
        :param model: if this is a data gathering agent, pass in the
            full model here and it will be used for action selection
        :param state_size: the size of the state space as the number of floats
        :param action_size: the size of the action space as the number
            of discrete actions that can be taken
        :param model: for a data gathering agent, a full trained (or
            partially trained) pytorch model
        :param seed: used to initialize a new model
        :param double_dqn: whether to use double DQN
        :param batch_size: how many episodes to sample per learning step
        :param buffer_size: how many episodes to keep in the replay
            buffer before discarding episodes
        :param hidden_size: this size of each of the two hidden layers
            of the neural network
        :param lr: the learning rate for the optimizier
        :param gamma: the discount factor for transfering future reward
            back down the chain
        :param tau: the rate at which we transfer the latest model
            weights to the target network
        :param device_name: cpu, cuda:0, etc.  if null, default to
            either cpu or cuda:0, used for running parallel simulations
            on a multi-GPU instance
        """
        if device_name is None:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.batch_size = batch_size
        self.per = False
        self.double_dqn = double_dqn
        self.is_beta = 0
        self.learn_c = 0
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.learn_every = learn_every
        if model is None:
            model = Model(state_size=state_size, action_size=action_size, seed=seed).to(self.device)
            local_model = Model(state_size=state_size, action_size=action_size, seed=seed).to(self.device)
        else:
            local_model = model
        self.local_model = local_model
        self.target_model = model
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        self.history = History()

    def act(self, state, eps):
        """
        :param state: a 37 wide tensor representing the current game state
        :param eps: the chance of choosing a purely random action instead of exploiting
        """
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        qvalues = self.local_model(state).detach().cpu().numpy()
        return np.argmax(qvalues)

    def step(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool):
        """ store a sample in the buffer, and do nothing else.  for
        distributed experience gathering, that's all that's necessary
        and learning is handled separately in the the learn method.
        :param state: the state which we determined this action from
        :param action: the action we took in this experience
        :param reward: the reward received from this exact point
        :param next_state: an array of floats representing the next state
        :param done: whether this is a terminal state for an episode,
            which signals that we can discard future Q values when
            calculating the expected value
        """
        sample = Sample(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.history.add_sample(sample)

    def learn(self):
        """ typically called after step(), every `learn_every` steps and
        once we have sufficient samples in our history, try to improve
        our behaviour model by taking a random sample and applying the
        iterative update algorithm
        """
        self.learn_c += 1
        if self.learn_c % self.learn_every == 0 and len(self.history) >= 1000:
            batch = self.history.random_sample(self.batch_size)
            self.learn_batch(batch)

    def learn_batch(self, batch: list[Sample]):
        """ take a batch of samples, calculate the TD error, and move
        the local network closer to the optimal function

        :param batch: a list of samples which we pulled from the replay buffer
        """
        self.optimizer.zero_grad()
        rewards = torch.Tensor([x.reward for x in batch]).to(self.device)
        actions = torch.LongTensor([x.action for x in batch]).to(self.device)
        next_states = torch.Tensor([x.next_state for x in batch]).to(self.device)
        states = torch.Tensor([x.state for x in batch]).to(self.device)
        with torch.no_grad():
            if self.double_dqn:
                # we get the best action from the local model
                target_q_next_actions = torch.argmax(self.local_model(next_states), 1).unsqueeze(1)
                # and index into the target model
                target_q_next = self.target_model(next_states).gather(1, target_q_next_actions).squeeze(1)
            else:
                target_q_next = torch.max(self.target_model(next_states), 1).values
            goal = rewards + GAMMA * target_q_next
        local_q = self.local_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # note, if importance sampling is used then instantiate as
        # nn.MSELoss(reduction='none') to access individual sample errors
        criterion = nn.MSELoss()
        loss = criterion(local_q, goal)
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        
    def soft_update(self):
        """
        move the target network slightly closer to the local network.
        the original algorithm synced the target model with the local
        model all at once periodically, while soft updates have a
        smoother transition.  the rate at which we adjust the target
        network is controlled by the `tau` parameter.
        """
        with torch.no_grad():
            for source, dest in zip(self.local_model.parameters(), self.target_model.parameters()):
                dest.copy_(dest * (1 - self.tau) + source * self.tau)
