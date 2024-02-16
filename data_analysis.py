# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.lineplot(data=data, x="K", y="Success", hue="Bot")
# plt.title("Success Rate vs. Number of Aliens")
# plt.xlabel("Number of Aliens (K)")
# plt.ylabel("Success Rate")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bot1 import bot1_move
from bot2 import bot2_move
from bot3 import bot3_move
# from bot4 import bot4_move  # Assume this exists

# Assuming run_simulation is a function that simulates the game 
# and returns a tuple (success: bool, steps: int)
def run_simulation(bot_function, D, K):
    # This function should be implemented to simulate the game
    # using the given bot function, ship size D, and number of aliens K
    # For simplicity, this is just a placeholder
    success = False  # Whether the bot succeeded
    steps = 0  # Number of steps taken
    return success, steps

def collect_data(D, K_values, num_runs):
    results = []
    bots = [bot1_move, bot2_move, bot3_move]  # Add bot4_move as needed

    for K in K_values:
        for bot_function in bots:
            for _ in range(num_runs):
                success, steps = run_simulation(bot_function, D, K)
                results.append({
                    "Bot": bot_function.__name__,
                    "K": K,
                    "Success": success,
                    "Steps": steps
                })

    return pd.DataFrame(results)

# Data Collection Configuration
D = 10  # Ship layout dimension
K_values = range(1, 11)  # Range of alien counts to test
num_runs = 100  # Number of runs per bot per K value

# Collecting Data
data = collect_data(D, K_values, num_runs)

# Data Analysis
print(data.groupby(["Bot", "K"]).mean().reset_index())

# Data Visualization
sns.lineplot(data=data, x="K", y="Success", hue="Bot", marker="o")
plt.title("Success Rate vs. Number of Aliens")
plt.xlabel("Number of Aliens (K)")
plt.ylabel("Success Rate")
plt.legend(title="Bot", title_fontsize='13', fontsize='12')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Visualization for Steps Taken
sns.lineplot(data=data, x="K", y="Steps", hue="Bot", marker="o", style="Bot", dashes=False)
plt.title("Average Steps Taken vs. Number of Aliens")
plt.xlabel("Number of Aliens (K)")
plt.ylabel("Average Steps Taken")
plt.legend(title="Bot", title_fontsize='13', fontsize='12')
plt.grid(True)
plt.tight_layout()
plt.show()
