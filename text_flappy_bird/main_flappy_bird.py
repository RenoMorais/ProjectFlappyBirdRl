# criando classe Q learning para usar no ambinete flappy bird
# reference: https://aspram.medium.com/learning-flappy-bird-agents-with-reinforcement-learning-d07f31609333

import os
import sys
import time
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import text_flappy_bird_gym
from tqdm import tqdm


class Agent:
    """The Base class that is implemented by other classes to avoid the duplicate 'act' method"""

    ## take random action if random float is less than epsilon
    ## otherwise select action with highest Q-score for the given state
    def act(self, state, train=True):  # epsilon-greedy policy
        if train and np.random.uniform(0, 1) < self.eps:
            action = self.env.action_space.sample()
            return action
        else:
            action = np.argmax(self.q[state])
            return action

    def train(self, num_episodes=100_000):
        all_reward_sums = []

        for episode in tqdm(range(self.num_episodes)):
            state, _ = self.env.reset()
            total_reward = 0

            while True:
                action = self.act(state)

                # Apply action and return new observation of the environment
                next_state, reward, done, _, info = self.env.step(action)

                # For SARSA acquiring the on-policy next action
                next_action = self.act(next_state)

                if done:
                    reward = -1

                # Update total reward
                total_reward += reward

                # Update q values table
                self.update(state, action, next_state, next_action, reward)

                state = next_state

                if done:
                    break

            # if episode % 100 == 0:
            #     print(f"\tEpisode: {episode} | Total Reward: {total_reward} | Score: {info['score']}")

            all_reward_sums.append(total_reward)

        return all_reward_sums

    def play(self, render=True, max_score=100):
        state, _ = self.env.reset()
        done = False

        while True:
            # Apply action and return new observation of the environment
            action = self.act(state, train=False)
            next_state, _, done, _, info = self.env.step(action)

            if render:
                os.system("clear")
                sys.stdout.write(self.env.render())
                time.sleep(0.2)

            if done:
                return info["score"]

            if info["score"] >= max_score:
                print(f"Game ended because it reached score={max_score}")
                return info["score"]

            # Update q values table
            state = next_state


class Q_Agent(Agent):
    """The Q-Learning Agent Class"""

    def __init__(self, env, eps=0.2, step_size=0.7, discount=0.95, num_episodes=10_000, q=None):
        self.env = env

        self.eps = eps
        self.step_size = step_size
        self.discount = discount

        self.num_episodes = num_episodes

        if q:
            self.q = q
        else:
            # The dict of action-value estimates.
            self.q = defaultdict(lambda: np.zeros(2))

    def update(self, state, action, next_state, next_action, reward):
        old_value = self.q[state][action]
        next_max = np.max(self.q[next_state])

        new_value = (1 - self.step_size) * old_value + self.step_size * (reward + self.discount * next_max)
        self.q[state][action] = new_value


class SarsaAgent(Agent):
    """The SARSA Agent Class"""

    def __init__(self, env, eps=0.2, step_size=0.7, discount=0.95, num_episodes=1_000_000, q=None):
        self.env = env

        self.eps = eps
        self.step_size = step_size
        self.discount = discount

        self.num_episodes = num_episodes

        if q:
            self.q = q
        else:
            # The dict of action-value estimates.
            self.q = defaultdict(lambda: np.zeros(2))

    def update(self, state, action, next_state, next_action, reward):
        self.q[state][action] += self.step_size * (
            reward + self.discount * (self.q[next_state][next_action]) - self.q[state][action]
        )


class RandomAgent(Agent):
    """The Random Agent Class"""

    def __init__(self, env, num_episodes, *args, **kwargs):
        self.env = env

        self.num_episodes = num_episodes

    def act(self, *args, **kwargs):
        return self.env.action_space.sample()

    def update(self, *args, **kwargs):
        pass


def main(num_episodes=10_000):
    env = gym.make("TextFlappyBird-v0", height=10, width=18, pipe_gap=4)

    agents = {
        "RandomAgent": RandomAgent(env, num_episodes),
        "Q-Learning": Q_Agent(env, eps=0.2, step_size=0.7, discount=0.95, num_episodes=num_episodes),
        # "Sarsa": SarsaAgent(env, eps=0.2, step_size=0.7, discount=0.95, num_episodes=num_episodes),
    }

    all_reward_sums = {}  # Contains sum of rewards during episode
    execution_times = {}
    scores = {}

    for algorithm, current_agent in agents.items():
        all_reward_sums[algorithm] = []

        print(f"Training algorithm: {algorithm}")
        start_time = time.time()
        all_reward_sums[algorithm] = current_agent.train(num_episodes=num_episodes)
        final_time = time.time() - start_time
        execution_times[algorithm] = final_time

        print(f"Testing algorithm: {algorithm}")
        score = current_agent.play()
        scores[algorithm] = score
        print(f"Testing Score: {score}")
        print()

    env.close()

    # Plot Reward
    for algorithm, current_agent in agents.items():
        print(f"Algorithm: {algorithm}")
        print(f"Trained in {execution_times[algorithm]} secs.")
        print(f"Test score: {scores[algorithm]}")
        print()
        plt.plot(all_reward_sums[algorithm], label=algorithm)

    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode", rotation=0, labelpad=40)
    plt.xlim(0, num_episodes)
    plt.ylim(0, 500)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
