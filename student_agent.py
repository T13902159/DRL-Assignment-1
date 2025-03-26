# student_agent.py
import numpy as np
import pickle
import os

# Load trained Q-table
q_table_path = "q_table.pkl"
if not os.path.exists(q_table_path):
    raise FileNotFoundError(
        "Error: q_table.pkl not found! Train your agent first.")

with open(q_table_path, "rb") as f:
    Q_table = pickle.load(f)

# Ensure Q-table is in the correct format
if isinstance(Q_table, dict):
    Q_table = np.array(list(Q_table.values()))


def get_action(obs):
    """
    Selects the best action using the trained Q-table.
    If the state is missing from the Q-table, chooses a random action.
    """
    if obs in range(Q_table.shape[0]):
        return np.argmax(Q_table[obs])  # Select the best action
    else:
        return np.random.choice([0, 1, 2, 3, 4, 5])  # Random fallback action
