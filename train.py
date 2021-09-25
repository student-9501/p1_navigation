import argparse
from banana_env import Environment
import numpy as np
from collections import deque
import torch
from agent import Agent
import os
import wandb


if os.environ.get('WANDB_ENTITY'):
    wandb.login()


class Wand:
    """ so the trainer can run without a wandb account """
    def __init__(self, agent):
        self.agent = agent

    def login(self):
        if not os.environ.get('WANDB_ENTITY'):
            return
        wandb.login()

    def init(self, **kwargs):
        if not self.configured:
            return
        wandb.init(**kwargs)

    def log(self, arg):
        if not self.configured:
            return
        wandb.log(arg)

    @property
    def configured(self):
        return os.environ.get('WANDB_ENTITY')


def main(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
         double_dqn=False, stop_score=None, env_id=None, project='deep-rl-banana',
         filename='Banana_Linux_NoVis/Banana.x86_64'):
    """
    :param n_episodes: number of training episodes
    :param max_t: maximum number of timesteps per episode
    :param eps_start: tarting value of epsilon, for epsilon-greedy action selection
    :param eps_end: minimum value of epsilon
    :param eps_decay: decay rate of epsilon per episode
    :param double_dqn: whether to enable double dqn learning
    :param stop_score: if set, stop as soon as the agent hits a threshold
        score averaged over 100 episodes
    :param project: the project name to log runs to in weights and biases
    :param env_id: if set, launch unity on a new port to run parallel sessions
    :param filename: the unity binary to run, eg. 'Banana_Linux_NoVis/Banana.x86_64'
    """
    env = Environment(env_id=env_id, filename=filename)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, double_dqn=double_dqn)
    wand = Wand(agent)
    wand.init()

    if os.environ.get('WANDB_ENTITY'):
        wand.init(project=project, entity=os.environ.get('WANDB_ENTITY'), group='final', config={
            'LR': agent.lr,
            'TAU': agent.tau,
            'HIDDEN_SIZE': agent.hidden_size,
            'BUFFER_SIZE': agent.buffer_size,
            'GAMMA': agent.gamma,
            'DOUBLE_DQN': agent.double_dqn,
            'PER': agent.per,
        })

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        wand.log({
            "scores": np.mean(scores_window),
            "eps": eps,
        })
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if stop_score is not None and np.mean(scores_window) >= stop_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    else:
        torch.save(agent.local_model.state_dict(), 'bananas-final.pth')
    print('done training')
    os._exit(0)  # workaround for unity cleanup code hanging sometimes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--double-dqn', dest='double_dqn', action='store_true', default=True)
    parser.add_argument('--no-double-dqn', dest='double_dqn', action='store_false')
    parser.add_argument('--env_id', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--filename', type=str, default='Banana_Linux_NoVis/Banana.x86_64')
    args = parser.parse_args()
    print(args)
    main(
        double_dqn=args.double_dqn,
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        env_id=args.env_id,
        filename=args.filename,
    )
