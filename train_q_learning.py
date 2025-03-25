import gym
import numpy as np
import pickle
import os

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3")

# Q-table initialization
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9   # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate per episode
episodes = 25000  # Total training episodes

# Training loop
for episode in range(episodes):
    state = env.reset()[0]  # Get initial state
    done = False

    while not done:
        # Choose action: Exploration vs. Exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random action)
        else:
            action = np.argmax(q_table[state])  # Exploit (best action)

        # Take action and observe result
        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma *
            np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state  # Move to the next state

    # Decay epsilon for better exploitation over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Reduce learning rate gradually
    alpha = max(0.01, alpha * 0.99)

    # Print progress every 1000 episodes
    if episode % 1000 == 0:
        print(
            f"Episode {episode}/{episodes}, Epsilon: {epsilon:.3f}, Alpha: {alpha:.3f}")

# Save trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as 'q_table.pkl'.")
