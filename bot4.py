# Implement Bot 4: A bot with a custom strategy.

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from queue import PriorityQueue
from ship_layout import generate_ship_layout

def simulate_bot4(D, K_range, num_trials, ship_layout):
    results = []
    success_count = 0

    for K in K_range:
        print(f"Simulation for {K} Aliens")
        for trial in range(num_trials):
            print(f"Trial: ({K}, {trial})")
            bot_position = random_position(D, ship_layout)
            captain_position = random_position(D, ship_layout)
            
            while captain_position == bot_position:
                captain_position = random_position(D, ship_layout)
            
            exclude_positions = [bot_position]
            remove_aliens(D, ship_layout)
            aliens_positions = place_aliens(D, ship_layout, K, exclude_positions)
            risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
            next_move, path, _ = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
            steps = 0
            success = False
            alive = True

            while steps < 1000:
                steps += 1
                bot_position = next_move
                # next_move, path, _ = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)

                if bot_position == captain_position:
                    success = True
                    break

                risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
                next_move, path, _ = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
                aliens_positions = move_aliens(aliens_positions, ship_layout)
                if bot_position in aliens_positions:
                    success = False
                    alive = False
                    break
                # visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, aliens_positions, path)
            survival = (alive and steps == 1000) / (num_trials - success_count) if num_trials - success_count != 0 else 1.0
            # Calculate survival rate as (bot didn't die and crossed 1000 steps) / (total trials - successful runs)
            # Instead of calculating averages and success rates here, simply record each trial's outcome
            results.append({
                'K_bot4': K,
                'Success_bot4': success,
                'Steps_bot4': steps,
                'Survival_bot4': survival
            })
    # Print statement after each simulation
    print(f"Simulation for K={K} is complete")

    return pd.DataFrame(results)


# def simulate_bot4(D, K_range, num_trials, ship_layout):
#     results = []

#     for K in K_range:
#         print(f"Simulation for {K} Aliens")
#         success_count = 0
#         steps_list = []
        
#         for trial in range(num_trials):
#             print(f"Trial: ({K}, {trial})")
#             bot_position = random_position(D, ship_layout)
#             captain_position = random_position(D, ship_layout)
#             exclude_positions = [bot_position]
#             remove_aliens(D, ship_layout)
#             aliens_positions = place_aliens(D, ship_layout, K, exclude_positions)
            
#             risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
#             next_move, path, _ = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
#             #bot_position, captain_position, ship_layout, aliens_positions 
#             steps = 0
#             success = False
            
            # while steps < 1000:
            #     steps += 1
            #     bot_position = next_move
            #     if bot_position == captain_position:
            #         success = True
            #         break
            #     aliens_positions = move_aliens(aliens_positions, ship_layout)
            #     risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
            #     next_move, path, _ = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
                
            #     if bot_position in aliens_positions:
            #         success = False
            #         break
                
            #     visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, aliens_positions, path)
            # success_count += 1 if success else 0
            # steps_list.append(steps)
            
    #     success_rate = success_count / num_trials
    #     avg_steps = np.mean(steps_list)
    #     results.append({'K': K, 'SuccessRate': success_rate, 'AverageSteps': avg_steps})
    
    # return pd.DataFrame(results)

def heuristic(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D, ship_layout):
    """Generate a random position within the grid bounds that is not blocked."""
    while True:
        x, y = random.randint(0, D-1), random.randint(0, D-1)
        if ship_layout[x, y] == 1:  # Assuming 1 indicates an open cell
            return (x, y)

def calculate_risk_scores(ship_layout, aliens_positions, risk_range=2):
    D = ship_layout.shape[0]
    risk_scores = np.zeros_like(ship_layout, dtype=np.float32)
    # Adjust risk increment based on proximity to aliens
    for alien in aliens_positions:
        for dx in range(-risk_range, risk_range + 1):
            for dy in range(-risk_range, risk_range + 1):
                nx, ny = alien[0] + dx, alien[1] + dy
                if 0 <= nx < D and 0 <= ny < D:
                    distance = np.sqrt(dx**2 + dy**2)
                    risk_increment = 1 / (1 + distance)  # Decaying function of distance
                    risk_scores[nx, ny] += risk_increment
    
    # Normalize risk scores to range [0, 1]
    max_risk = np.max(risk_scores)
    if max_risk > 0:
        risk_scores /= max_risk
    return risk_scores

def find_path_with_risk_assessment(start, goal, ship_layout, risk_scores, risk_multiplier=2):
    D = ship_layout.shape[0]
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from.get(current)  # Removed , None) -------------
            path.reverse()  # Reverse to get path from start to goal
            return path[0:]
            # break

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < D and 0 <= next[1] < D and ship_layout[next[0], next[1]] == 1:  # Assume 0 as blocked changed !=0 to ==1 -------
                risk_scores = np.array(risk_scores)  # Ensure risk_scores is a NumPy array
                new_cost = cost_so_far[current] + 1 + (risk_scores[next[0], next[1]] * risk_multiplier)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    open_set.put((priority, next))
                    came_from[next] = current
    return []

def visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, aliens_positions, path):  # Added path ------
    """Visualize the ship layout with risk scores, bot, captain, and aliens."""
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis  # Use a colormap that represents risk well, like viridis
    risk_overlay = np.ma.masked_where(ship_layout == 0, risk_scores)  # Mask areas with no risk
    ax.imshow(ship_layout, cmap='Greys', interpolation='nearest')  # Ship layout in grey scale
    ax.imshow(risk_overlay, cmap=cmap, alpha=0.5, interpolation='nearest')  # Overlay risk scores

    # # Highlighting the path
    # for p in path:
    #     ax.plot(p[1], p[0], 'yx')
    if isinstance(path, list) and all(isinstance(p, tuple) and len(p) == 2 for p in path):
        for p in path:
            ax.add_patch(plt.Circle((p[1], p[0]), 0.3, color='yellow'))  # Correctly visualize path

    # Mark bot, captain, and aliens
    ax.plot(bot_position[1], bot_position[0], 'bo')  # Bot in blue
    ax.plot(captain_position[1], captain_position[0], 'go')  # Captain in green
    for alien in aliens_positions:
        ax.plot(alien[1], alien[0], 'rx')  # Aliens in red with an 'x'

    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    plt.show()

def place_aliens(D, grid, count, exclude_positions):
    aliens = []
    while len(aliens) < count:
        position = random_position(D, grid)
        if position not in exclude_positions:
            aliens.append(position)
            grid[position] = 2  # Assuming '2' marks an alien, adjust as needed
    return aliens

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def remove_aliens(D, grid):
    for x in range(D):
        for y in range(D):
            grid[x, y] = 1 if grid[x,y] != 0 else 0

def move_aliens(alien_positions, grid):
    new_alien_positions = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible directions: right, down, left, up
    for pos in alien_positions:
        valid_moves = [((pos[0] + d[0], pos[1] + d[1])) for d in directions if 0 <= pos[0] + d[0] < grid.shape[0] and 0 <= pos[1] + d[1] < grid.shape[1] and grid[pos[0] + d[0], pos[1] + d[1]] == 1]
        if valid_moves:
            new_pos = random.choice(valid_moves)  # Choose a random valid move
            new_alien_positions.append(new_pos)
        else:
            new_alien_positions.append(pos)  # Stay in place if no valid move
    return new_alien_positions

def bot4_move(bot_position, captain_position, ship_layout, aliens_positions):
    """Calculate the next move for Bot 4 considering risk scores."""
    risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
    # reconstruct = reconstruct_path(path, bot_position, captain_position)
    path = find_path_with_risk_assessment(bot_position, captain_position, ship_layout, risk_scores)
    # Ensure there's a path and it has at least one step beyond the current position
    next_step = path[1] if len(path) > 1 else bot_position
    return next_step, path, risk_scores  # Return the next step, full path for visualization, and risk scores
    # return path[0] if path else bot_position

# Example usage placeholder
if __name__ == "__main__":
    D = 30
    ship_layout = generate_ship_layout(D)
    K_range = range(0,101,2)
    num_trials = 250

    bot4_data = {'K_bot4': [], 'Success_bot4': [], 'Steps_bot4': [],'Survival_bot4': []}

    bot4_df = simulate_bot4(D, K_range, num_trials, ship_layout)

    print(bot4_df.head())
    print(bot4_df.describe())

    bot4_df.to_csv('bot4_data.csv', index=False)

    # Calculate success rate and average steps for plotting
    #plot_data = bot4_df.groupby('K').agg(SuccessRate=('Success', 'mean'), Survival Rate=('Steps', 'mean')).reset_index()

    # Plotting Success Rate vs Number of Aliens
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bot4_df, x='K_bot4', y='Success_bot4')
    plt.title('Bot 4 Success Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Success Rate')
    plt.show()

    # Plotting Average Steps vs Number of Aliens
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bot4_df, x='K_bot4', y='Survival_bot4', color='red')
    plt.title('Bot 4 Survival Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Survival Rate to Reach the Captain')
    plt.show()

    # D = random.randint(1, 50)  # Dynamic ship size
    # ship_layout = generate_ship_layout(D)
    # print(ship_layout.shape)
    # print(ship_layout)
    # bot_position = random_position(D, ship_layout)
    # captain_position = random_position(D, ship_layout)

    # alien_count = random.randint(1, D//2)
    # exclude_positions = [bot_position]
    # #, captain_position]
    # alien_positions = place_aliens(D, ship_layout, alien_count, exclude_positions)  # Reuse or define this function
    # #print(alien_positions)
    # while captain_position == bot_position:
    #     captain_position = random_position(D, ship_layout)
    
    # # Correctly calculate risk scores before finding the path
    # risk_scores = calculate_risk_scores(ship_layout, alien_positions)
    # next_move, path, risk_scores = bot4_move(bot_position, captain_position, ship_layout, alien_positions)
    # bot_position = next_move
    # print("Path:", path)

    # steps = 0
    # max_steps = 1000 # To Prevent Infinite Loop
    # while bot_position != captain_position and steps < max_steps:
    #     steps += 1
    #     alien_positions = move_aliens(alien_positions, ship_layout)  # Move aliens
    #     previous_position = bot_position  
              
    #     # Recalculate risk scores based on updated alien positions
    #     # risk_scores = calculate_risk_scores(ship_layout, alien_positions)
    #     # Find the new path with updated risk scores and alien positions
    #     next_move, path, risk_scores = bot4_move(bot_position, captain_position, ship_layout, alien_positions)
    #     path = next_move
    #     bot_position = next_move 
    #     # if next_move else bot_position  # Update bot position only if next_move is valid

    #     if bot_position in alien_positions:
    #         print(f"Mission Failed(1) : Captured by Aliens at {bot_position} on step {steps}")
    #         break
    #     alien_positions = move_aliens(alien_positions,ship_layout)
    #     if bot_position in alien_positions:
    #         print(f"Mission Failed(1) : Captured by Aliens at {bot_position} on step {steps}")
    #         break
    #     if bot_position == previous_position:
    #         print(f"Step {steps}: Bot is stuck at {bot_position}, trying to reach Captain at {captain_position}.")
    #     else:
    #         print(f"Step {steps}: Bot moved to {bot_position}, aiming for Captain at {captain_position}.")

    #     if bot_position == captain_position:
    #         print("Bot has successfully reached the captain!")
    #         break

    #     # Update the visualization at each step
    #     if steps % 3 == 0:  # For example, update the visualization every 10 steps
    #         visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, alien_positions, path)
