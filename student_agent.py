import numpy as np
import pickle
import os

# Load trained Q-table
q_table_path = "q_table.pkl"
if os.path.exists(q_table_path):
    with open(q_table_path, "rb") as f:
        Q_table = pickle.load(f)
else:
    raise FileNotFoundError(
        "Error: Q-table file not found. Train your agent first!")


def get_action(obs):
    """
    Selects the best action using the trained Q-table.
    If the state is missing from the Q-table, chooses a random action.
    """
    if obs in Q_table:
        return np.argmax(Q_table[obs])  # Best action from the Q-table
    else:
        return np.random.choice([0, 1, 2, 3, 4, 5])  # Random fallback action
