import argparse
import math
from datetime import datetime

# import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

# from telegram_notification import begin_notification, end_notification


def parse_args():
    parser = argparse.ArgumentParser(
        description="RL Agent for Flappy Bird.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train algorithm",
    )

    parser.add_argument(
        "--test", type=str, dest="test_filepath", help="Test algorithm on the trained model saved in `TEST_FILEPATH`"
    )

    # parser.add_argument("--plot", type=str, dest="plot_filepath", help="Test algorithm.")
    parser.add_argument("--plot", type=str, nargs="+", dest="plot_filepaths", help="Evaluations files")
    parser.add_argument(
        "-n",
        type=int,
        dest="n_steps",
        default=100_000_000,
        help="Nb. timesteps used in training (only used with the --plot option)",
    )

    args = parser.parse_known_args()[0]
    return args


def make_env(env_id: str, rank: int, seed: int = 0, mode: str = "train"):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = gym.make(env_id, audio_on=False, render_mode="rgb_array", use_lidar=False, mode=mode)
        env.reset(seed=seed + rank)

        return env

    set_random_seed(seed)
    return _init


def train():
    env_id = "FlappyBird-v0"
    num_cpu = 4  # Number of processes to use

    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i, mode="train_random") for i in range(1, num_cpu + 1)])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_final_eps=0.01,
        target_update_interval=250,
        buffer_size=50_000,
        batch_size=32,
        tensorboard_log="./tensorboard/",
    )

    # Save a checkpoint every 100_000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path="./logs/", name_prefix="rl_model", save_replay_buffer=True, save_vecnormalize=True
    )

    # Use deterministic actions for evaluation
    eval_env = gym.make(env_id, audio_on=False, render_mode="rgb_array", use_lidar=False, mode="eval", seed=0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=100_000 / num_cpu,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # timesteps = int(100_000_000)
    timesteps = int(10_000_000)

    failed = False
    try:
        # Evaluate before training
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"[BEFORE] mean_reward={mean_reward:.2f} +/- {std_reward}")

        # Train the agent
        # start_date = datetime.now()
        # begin_notification(start_date)

        model.learn(
            total_timesteps=timesteps,
            log_interval=1_000,
            progress_bar=True,
            callback=callback,
            tb_log_name="seed_train_random_100M_steps",
        )
        env.close()
    except Exception:
        failed = True
    finally:
        # end_notification(start_date, failed)

        # Save the agent
        model.save("dqn_flappy_bird")

        del model
        model = DQN.load("dqn_flappy_bird")

        # Evaluate the trained agent
        # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        # print(f"[AFTER] mean_reward={mean_reward:.2f} +/- {std_reward}")
        eval_env.close()


def play_game(env, model):
    obs, _ = env.reset(seed=42)

    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        total_reward += reward

        if done:
            print("Score: ", info["score"])
            print(f"Total Reward: {total_reward: .2f}")
            print()
            break
    env.close()

    return info["score"]


def test(test_filepath, n_tries=1):
    model = DQN.load(test_filepath)

    # Evaluate the trained agent
    # eval_env = gym.make("FlappyBird-v0", audio_on=False, render_mode="human", use_lidar=False, mode="eval")
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    scores = []
    for i in range(n_tries):
        print(f"Playing try number {i+1}")
        play_env = gym.make("FlappyBird-v0", audio_on=False, render_mode="human", use_lidar=False, mode="eval", seed=0)
        score = play_game(play_env, model)
        scores.append(score)

    print("Mean Scores: ", np.mean(scores))
    print("Std Scores: ", np.std(scores))


def moving_average(y, window_size=10):
    average_y = []

    for ind in range(len(y) - window_size + 1):
        average_y.append(np.mean(y[ind : ind + window_size]))

    for ind in range(window_size - 1):
        average_y.insert(0, np.nan)

    return average_y


def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor(math.log10(abs(value)) / 3)
    value = round(value / 1000**num_thousands, 2)
    return f"{value:g}" + " KMGTPEZY"[num_thousands]


def plot_moving_average(x, y, n_steps=100_000_000, eval_freq=100_000):
    y_data = [v for _, v in y.items()]

    window_size = int(max(10, ((n_steps / eval_freq) / 10)))
    # average_y = moving_average(y, window_size)
    average_y = [moving_average(yy, window_size) for yy in y_data]

    plt.figure(figsize=(13, 5))
    # plt.plot(x, y, "k.-", label="Reward total")
    # plt.plot(x, average_y, "r.-", label=f"Média móvel de {window_size} timesteps")

    for i, plot_name in enumerate(y.keys()):
        label_name = "train_random" if "train_random" in plot_name else "train"
        plt.plot(x[plot_name], average_y[i], ".-", label=label_name)

    step = int(n_steps / 10)
    x_ticks = range(0, n_steps + step, step)
    x_labels = [format_func(t) for t in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.xlim(eval_freq, n_steps + eval_freq)

    plt.xlabel("Timestep")
    plt.ylabel("Reward total", rotation=0, labelpad=40)

    plt.grid(linestyle=":")
    plt.legend(title="Posição inicial")
    plt.title(f"Etapa de avaliação - Média móvel do reward total dos últimos {window_size} timesteps")

    plt.savefig(f"evaluation_{format_func(n_steps)}_steps.png")
    plt.show()


# def plot_learning_curve(x, y):
#     plt.plot(x, y, "o-")
#     # for t, v in zip(data["timesteps"], mean_rewards):
#     #     plt.text(t + 2500, v, f"{v:.2f}", ha="left")

#     x_labels = [format_func(t) for t in x]
#     plt.xticks(x, x_labels, rotation=45)

#     plt.xlabel("Timestep")
#     plt.ylabel("Reward total", rotation=0, labelpad=40)

#     plt.grid(axis="x", color="0.95")
#     plt.show()


def evaluate_learning_curve(plot_filepaths, n_steps=100_000_000):
    timesteps_data = {}
    results = {}
    for filepath in plot_filepaths:
        data = np.load(filepath)

        mean_rewards = []
        for total_rewards in data["results"]:
            mean_rewards.append(np.mean(total_rewards))

        timesteps_data[filepath] = data["timesteps"]
        results[filepath] = mean_rewards

    plot_moving_average(timesteps_data, results, n_steps)

    # plot_learning_curve(data["timesteps"], mean_rewards)


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        train()
    elif args.test_filepath:
        test(args.test_filepath)
        # test("dqn_flappy_bird_bkp")
    elif args.plot_filepaths:
        evaluate_learning_curve(args.plot_filepaths, args.n_steps)
        evaluate_learning_curve(args.plot_filepaths, args.n_steps)
