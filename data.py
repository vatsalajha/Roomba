import pandas as pd
from functools import partial

# Example bot move functions for simplicity
from bot1 import bot1_move
from bot2 import bot2_move
from bot3 import bot3_move
# from bot4 import bot4_move  # Assume this exists

# Modify run_simulation to return both outcome and steps
# Assume it's done

def collect_data(D, K_values, num_runs):
    results = []
    bots = [bot1_move, bot2_move, bot3_move]  # Add bot4_move as needed

    for K in K_values:
        for bot_move in bots:
            for _ in range(num_runs):
                success, steps = run_simulation(bot_move, D, K)
                results.append({
                    "Bot": bot_move.__name__,
                    "K": K,
                    "Success": success,
                    "Steps": steps
                })

    return pd.DataFrame(results)

# Configuration
D = 10  # Ship layout dimension
K_values = range(1, 11)  # Range of alien counts to test
num_runs = 100  # Number of runs per bot per K value

# Data Collection
data = collect_data(D, K_values, num_runs)

# Example analysis
print(data.groupby(["Bot", "K"]).mean())  # Average success rate and steps taken
