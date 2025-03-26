import gym
import numpy as np
import pickle
import os

# Load the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")  # Use "human" for GUI rendering

# Load trained Q-table
q_table_path = "q_table.pkl"
if not os.path.exists(q_table_path):
    raise FileNotFoundError(
        "Error: q_table.pkl not found! Train your agent first.")

with open(q_table_path, "rb") as f:
    Q_table = pickle.load(f)


def evaluate_agent(episodes=50):
    """
    Runs the trained agent for multiple episodes and computes the average score.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action = np.argmax(
                Q_table[state]) if state in Q_table else env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            step_count += 1

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1}: Reward = {episode_reward}, Steps Taken = {step_count}")

    avg_score = np.mean(total_rewards)
    print(f"\n✅ Average Score over {episodes} episodes: {avg_score:.2f}")

    return avg_score


if __name__ == "__main__":
    evaluate_agent()
