import time
import traceback

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import flappy_bird_gymnasium
from flappy_bird_gymnasium.agents.agent import Agent
from flappy_bird_gymnasium.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Feed Forward Neural Network."""

    def __init__(self):
        """Initializes the network architecture."""
        super(MLP, self).__init__()

        self.linear1 = nn.Sequential(nn.Linear(12, 64), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True))

        # self.linear1 = nn.Linear(3, 50)
        # self.linear2 = nn.Linear(50, 20)
        # self.linear3 = nn.Linear(20, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (Tensor): An input vector.

        Returns:
            Tensor: An output vector.
        """
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class MLPAgent(Agent):
    def __init__(self, render_mode="rgb_array", replay_buffer_size=100_000):
        self.env = gymnasium.make(
            "FlappyBird-v0",
            audio_on=False,
            render_mode=render_mode,
            use_lidar=False,
        )

        # Init model
        self.model = MLP().to(device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def act(self, state, eps=0.1):
        if np.random.uniform() > eps:
            state = torch.tensor(state).float().unsqueeze(0).to(device)

            self.model.eval()
            with torch.no_grad():
                action_values = self.model(state)
            self.model.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(0, self.env.action_space.n)

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        net_state_dict = self.net.state_dict()

        # Update the parameters in the target network
        for key in net_state_dict:
            target_net_state_dict[key] = net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def train(
        self,
        num_episodes=500_000,
        batch_size=32,
        discount_factor=0.99,
        exploration_steps=50_000,
        learning_rate=1e-6,  # 0.1
    ):
        # Set the exploration rate
        eps = eps_start = 0.75
        eps_end = 0.001
        eps_decay = 0.99995

        total_steps = 0

        best_reward = -9999999
        best_score = -1

        try:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Run the training loop
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                # print("State: ", state)

                total_reward = 0

                if total_steps > exploration_steps:
                    eps = max(eps_end, eps_decay * eps)

                actions = []

                while True:
                    # import ipdb

                    # ipdb.set_trace()

                    # Select an action according to the predicted Q value
                    if total_steps < exploration_steps:
                        action = np.random.randint(0, 2)
                    else:
                        action = self.act(state, eps)
                    # action = self.act(state, eps)
                    actions.append(action)

                    # Interact with the environment
                    next_state, reward, done, _, info = self.env.step(action)
                    # print(f"State: {state} | Action: {action} | Reward: {reward} | Done: {done} | Eps: {eps}")

                    # Store the experience in the replay buffer
                    self.replay_buffer.append(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )

                    state = next_state

                    # time.sleep(1 / 60)

                    if total_steps > exploration_steps:
                        # if render:
                        #     time.sleep(1 / 60)  # FPS

                        # if total_steps > 1000:
                        # if (
                        #     total_steps % update_every == 0
                        #     and len(self.replay_buffer) > batch_size
                        # ):
                        # Sample random minibatch of transition from replay buffer
                        minibatch = self.replay_buffer.sample(batch_size)

                        (
                            state_batch,
                            action_batch,
                            reward_batch,
                            next_state_batch,
                            done_batch,
                        ) = minibatch

                        # Get max predicted Q values (for next states) from local model
                        # Follow greedy policy: use the one with the highest value
                        Q_targets_next = self.model(next_state_batch).detach().max(1)[0].unsqueeze(1)

                        # Compute Q targets for current states
                        Q_targets = reward_batch + (discount_factor * Q_targets_next * (1 - done_batch))

                        # Get expected Q values from local model
                        Q_expected = self.model(state_batch).gather(1, action_batch.view(-1, 1))

                        # Compute loss
                        loss = F.mse_loss(Q_expected, Q_targets)

                        # Minimize the loss

                        ## Zero out the previously calculated gradients
                        optimizer.zero_grad()

                        ## Calculate gradients
                        loss.backward()

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                        ## Update parameters based on gradients
                        optimizer.step()

                    # Update the state and the score
                    # state_img_batch = new_state_img_batch
                    total_reward += reward

                    total_steps += 1

                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_score = info["score"]

                    if (episode + 1) % 1000 == 0:
                        output_filepath = f"trained_models/flappy_bird_{episode + 1}"
                        torch.save(self.model, output_filepath)

                    if done:
                        break

                print(
                    # f"Iteration: {episode+1:10}/{num_episodes} | Last Action: {action} | Reward: {total_reward: .8f} | Epsilon: {eps:.8f} | Score: {info['score']} | Actions: {actions}"
                    f"Iteration: {episode+1:10}/{num_episodes} | Last Action: {action} | Reward: {total_reward: .8f} | Epsilon: {eps:.8f} | Score: {info['score']}"
                )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(traceback.print_exc())
        finally:
            print(f"Best Reward: {best_reward} | Best Score: {best_score}")

            import ipdb

            ipdb.set_trace()

        output_filepath = f"trained_models/flappy_bird"
        torch.save(self.model, output_filepath)

        self.env.close
