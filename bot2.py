# bot2.py
# Implements Bot 2: Re-plans the shortest path at every step, considering current alien positions.

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from queue import PriorityQueue
from ship_layout import generate_ship_layout  # Ensure these are defined or imported

def simulate_bot2(D, K_range, num_trials, ship_layout):
    results = []
    success_count = 0
    for K in K_range:
        print(f"Simulation for {K} Aliens")
        for _ in range(num_trials):
            print(f"Trial: ({K}, {_})")
            #ship_layout = generate_ship_layout(D)
            bot_position = random_position(D, ship_layout)
            captain_position = random_position(D, ship_layout)
            exclude_positions = [bot_position]  # Exclude bot's position from alien placement
            remove_aliens(D, ship_layout)
            aliens = place_aliens(D, ship_layout, K, exclude_positions)

            while captain_position == bot_position:  # Ensure bot and captain are not in the same position
                captain_position = random_position(D, ship_layout)

            path = find_shortest_path(bot_position, captain_position, ship_layout, aliens)
            success = False
            alive = True
            steps_taken = len(path) if path else 0

            if path:
                for steps in range(1000):
                    if steps == len(path):
                        break
                    bot_position = path[steps]
                    if bot_position in aliens:
                        #print(f"Mission Failed(1) : Captured by Aliens at {bot_position} on step {steps}")
                        alive = False
                        success = False  # Set success to False when the bot encounters aliens
                        break
                    aliens = move_aliens(aliens, ship_layout)
                    #exclude_positions = [bot_position]  # Update exclude_positions after the bot moves
                    if bot_position in aliens:
                        #print(f"Mission Failed(2) : Captured by Aliens at {bot_position} on step {steps}")
                        alive = False 
                        success = False  # Set success to False when the bot encounters aliens
                        break
                    #print(f"Step {steps}: Bot at {bot_position}, Captain at {captain_position}")
                    if captain_position == bot_position:
                        #print(f"Mission Successful : Captain Saved on step {steps}")
                        success = True  # Set success to True when the bot reaches the captain
                        success_count += 1
                        break
                    #visualize_layout(ship_layout,path, bot_position, captain_position, aliens)
                # else:
                #     #print("Mission Failed(3) : No Path Found")
                #     success = False  # Set success to False if the bot doesn't reach the captain within the specified steps
            # else:
            #     #print("Mission Failed(4) : No Path Found")
            #     success = False  # Set success to False if there's no path from bot to captain

            # if path:
            #     for step_pos in path[1:]:
            #         if tuple(step_pos) in aliens:
            #             success = False
            #             break
            #         aliens = move_aliens(aliens, ship_layout)
            #     else:
            #         if tuple(path[-1]) == tuple(captain_position):
            #             success = True
            survival = (alive and steps_taken == 1000) / (num_trials - success_count) if num_trials - success_count != 0 else 1.0

            results.append({
                'K_bot2': K,
                'Success_bot2': success,
                'Steps_bot2': steps_taken,
                'Survival_bot2': survival
            })
    # Print statement after each simulation
    print(f"Simulation for K={K} is complete")

    return pd.DataFrame(results)


def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D, grid):
    """Generate a random position within the grid bounds that is not blocked."""
    possible_indices = []
    for y in range(D):
        for x in range(D):
            if grid[x, y] == 1:
                possible_indices.append((x,y))
    if not possible_indices:
        print("No indices available. Lower K")
    return random.choice(possible_indices)
    #return random.randint(0, D-1), random.randint(0, D-1)

def get_neighbors(position, grid, alien_positions):
    """Returns the valid neighbors for a position in the grid, excluding alien positions."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for direction in directions:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            if grid[neighbor] == 1 and neighbor not in alien_positions:  # Ensure the neighbor is an open cell and not occupied by an alien
                neighbors.append(neighbor)
    return neighbors

# Should we remove this now ???? - PATH PRINTING FUNCTION
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Find the shortest path from start to goal using A* pathfinding."""
def find_shortest_path(start, goal, grid, alien_positions):
    open_set = PriorityQueue()
    # Storing the nodes to be explored
    open_set.put((0, start))
    came_from = {start: None}
    # cost_so_far = {start: 0}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current = open_set.get()[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path[0:]  # No excluding -- Return path excluding the start position
        
        for neighbor in get_neighbors(current, grid, alien_positions):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == q_item[1] for q_item in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
                    
    return []  # If no path found
    # pass

def bot2_move(bot_position, captain_position, alien_positions, ship_layout):
    path = find_shortest_path(bot_position, captain_position, ship_layout, alien_positions)
    # reconstruct = reconstruct_path(path, bot_position, captain_position)
    print("Path:")
    print(path)
    if path and len(path) > 1: 
        next_step = path[1] # Made this 1 instead of 0th position in order to avoid repeating
    else:
        next_step = bot_position  # Stay in place if no path is found
    return next_step

def visualize_layout(layout, path, bot_position=None, captain_position=None, alien_positions=[]):
    fig, ax = plt.subplots()
    # inverted_layout = 1 - layout
    ax.imshow(layout, cmap='binary', interpolation='nearest')  # 'binary' for black and white
    # 0 white (BLOCKED), 1 black (OPEN)

    # Highlighting the path
    for p in path:
        ax.plot(p[1], p[0], 'yx')

    if bot_position:
        ax.plot(bot_position[1], bot_position[0], 'bo')  # Bot in blue
    if captain_position:
        ax.plot(captain_position[1], captain_position[0], 'go')  # Captain in green
    for alien in alien_positions:
        ax.plot(alien[1], alien[0], 'ro')  # Aliens in red

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

# Example usage placeholder
if __name__ == "__main__":
    #D = random.randint(1, 50)  # Dynamic ship size
    D = 30
    ship_layout = generate_ship_layout(D)
    print(ship_layout.shape)
    # print(ship_layout)
    # bot_position = random_position(D, ship_layout)
    # captain_position = random_position(D, ship_layout)

    # alien_count = random.randint(1, D//2)
    K_range = range(0, 101, 2)
    num_trials = 250

    bot2_data = {'K_bot2': [], 'Success_bot2': [], 'Steps_bot2': [],'Survival_bot2': []}
    
    # Pass pre-generated ship layout, bot position, and captain position to the simulation method
    bot2_df = simulate_bot2(D, K_range, num_trials, ship_layout)
     # Print or visualize results as needed
    print(bot2_df.head())  # Example: Print the first few rows of the data
    #print(bot2_df.describe())  # Example: Print summary statistics of the data

    # Save data to CSV for later use
    bot2_df.to_csv('bot2_data.csv', index=False)


    plt.figure(figsize=(10, 6))
    #sns.lineplot(data=data, x='K', y='Success', estimator=np.mean)
    sns.lineplot(data=bot2_df, x='K_bot2', y='Success_bot2')
    plt.title('Bot 2 Success Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Success Rate')
    plt.show()

    plt.figure(figsize=(10, 6))
    # sns.lineplot(data=data, x='K', y='Steps', estimator=np.mean)
    sns.lineplot(data=bot2_df, x='K_bot2', y='Survival_bot2')
    plt.title('Bot 2 Survival Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Survival Rate')
    plt.show()
    # visualize_layout(ship_layout, bot_position, captain_position, aliens)
        # Figure out visualisation in this case ##
    # next_move = bot2_move(bot_position, captain_position, alien_positions, ship_layout)
    # print(f"Bot starts at: {bot_position}, Captain at: {captain_position}")    
    # print(f"Bot 2 moves to: {next_move}")
    
    #visualize_layout(ship_layout, bot_position, captain_position, alien_positions)