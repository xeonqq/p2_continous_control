import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment
from ddpg_agent import DDPG_Agent


def get_next_state(env_info):
    next_state = env_info.vector_observations[0]  # get the next state
    return next_state


def get_reward(env_info):
    reward = env_info.rewards[0]  # get the reward
    return reward


def get_done(env_info):
    done = env_info.local_done[0]  # see if episode has finished
    return done


def get_env_step_results(env_info):
    return get_next_state(env_info), get_reward(env_info), get_done(env_info)


class Environment(object):
    def __init__(self, env):
        self._env = env
        self._brain_name = env.brain_names[0]
        self._brain = env.brains[self._brain_name]
        self._env_info = env.reset(train_mode=True)[self._brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self._env_info.agents))

        # number of actions
        self._action_size = self._brain.vector_action_space_size
        print('Number of actions:', self._action_size)

        # examine the state space
        state = self._env_info.vector_observations[0]
        print('States look like:', state)
        self._state_size = len(state)
        print('States have length:', self._state_size)
        self._agent = DDPG_Agent(self._state_size, self._action_size)

    def run_model(self, actor_model, num_episode=3, steps_per_episode=300):
        self._agent.load_model(torch.load(actor_model))

        for i in range(num_episode):
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            state = get_next_state(env_info)
            score = 0
            for j in range(steps_per_episode):
                action = self._agent.act(state, False)
                env_info = self._env.step(action)[self._brain_name]  # send the action to the environment
                next_state, reward, done = get_env_step_results(env_info)
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                time.sleep(1 / 30.0)
                if done:
                    print("Episode: {}, score: {}".format(i, score))
                    break

    def close(self):
        self._env.close()

    def train(self, min_score, n_episodes=200, max_t=1000):
        """Deep Q-Learning.
            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        for i_episode in range(1, n_episodes + 1):
            env_info = self._env.reset(train_mode=True)[self._brain_name]
            self._agent.reset()
            state = get_next_state(env_info)
            score = 0
            for t in range(max_t):
                action = self._agent.act(state, False)
                env_info = self._env.step(action)[self._brain_name]  # send the action to the environment
                next_state, reward, done = get_env_step_results(env_info)
                self._agent.step(state, action, reward, next_state, done)
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= min_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                torch.save(self._agent._actor_target.state_dict(), 'actor.pth')
                break
        return scores


def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("ddpg_score.png")
    plt.show()


def train(min_score):
    env = Environment(UnityEnvironment(file_name='Reacher_Linux_one_agent/Reacher.x86_64'))
    scores = env.train(min_score)
    plot_scores(scores)
    env.close()


if __name__ == "__main__":
    train(min_score=30)
