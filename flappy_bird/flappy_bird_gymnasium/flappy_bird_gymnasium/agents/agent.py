import time

import gymnasium
import numpy as np
from tqdm import tqdm

import flappy_bird_gymnasium


class Agent(object):
    """The Base class that is implemented by other classes to avoid the duplicate 'act' method"""

    ## take random action if random float is less than epsilon
    ## otherwise select action with highest Q-score for the given state
    def act(self, state, eps=0.2, train=True):  # epsilon-greedy policy
        if train and np.random.uniform(0, 1) < eps:
            action = self.env.action_space.sample()
            return action
        else:
            action = np.argmax(self.q[state])
            return action

    def train(self, num_episodes=1_000, render=True):
        self.env = flappy_bird_gym.make("FlappyBird-v0")

        best_reward = -9999999
        best_score = -1

        all_reward_sums = []

        try:
            # Set the exploration rate
            eps = eps_start = 0.2
            # eps_end = 0.01
            # eps_decay = 0.995

            for episode in tqdm(range(num_episodes)):
                state = self.env.reset()

                total_reward = 0

                # eps = max(eps_end, eps_decay * eps)

                while True:
                    if render:
                        self.env.render()

                    action = self.act(state, eps)

                    # Apply action and return new observation of the environment
                    next_state, reward, done, info = self.env.step(action)

                    # print("Q: ", self.q)

                    # For SARSA acquiring the on-policy next action
                    next_action = self.act(next_state, eps)

                    # if done:
                    #     reward = -10

                    # Update total reward
                    total_reward += reward

                    # Update q values table
                    self.update(state, action, next_state, next_action, reward)

                    state = next_state

                    if render:
                        time.sleep(1 / 60)  # FPS

                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_score = info["score"]

                    if done:
                        if render:
                            self.env.render()
                            time.sleep(0.5)
                        break

                # if episode % 100 == 0:
                #     print(f"\tEpisode: {episode} | Total Reward: {total_reward} | Score: {info['score']}")
                print(f"\t Reward: {total_reward}, Epsilon: {eps}, Score: {info['score']}")

                all_reward_sums.append(total_reward)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"Best Reward: {best_reward} | Best Score: {best_score}")
            print("Q: ", self.q)

            import ipdb

            ipdb.set_trace()

        self.env.close()

        return all_reward_sums
