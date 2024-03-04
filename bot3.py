# bot3.py
# Implements Bot 3 - At every time step, the bot re-plans the shortest path to the Captain, avoiding the current alien
# positions and any cells adjacent to current alien positions, if possible, then executes the next step in that plan.
# If there is no such path, it plans the shortest path based only on current alien positions, then executes the next
# step in that plan. Note: Bot 3 resorts to Bot 2 behavior in this case

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

from queue import PriorityQueue
from ship_layout import generate_ship_layout  # Ensure these are defined or imported

def simulate_bot3(D, K_range, num_trials, ship_layout):
    results = []
    success_count = 0
    #ship_layout = generate_ship_layout(D)

    for K in K_range:
        print(f"Simulation for {K} Aliens")
        for _ in range(num_trials):
            print(f"Trial: ({K}, {_})")
            bot_position = random_position(D, ship_layout)  # Place the bot randomly
            captain_position = random_position(D, ship_layout)  # Place the captain randomly
            exclude_positions = [bot_position]  # Exclude bot's position from alien placement
            remove_aliens(D, ship_layout)
            aliens = place_aliens(D, ship_layout, K, exclude_positions)  # Place aliens randomly

            while captain_position == bot_position:  # Ensure bot and captain are not in the same position
                captain_position = random_position(D, ship_layout)

            path = find_shortest_path(bot_position, captain_position, ship_layout, aliens, True)
            success = False
            alive = True
            steps_taken = len(path) if path else 0

            if path:
                for steps in range(1000):
                    if steps == len(path):
                        break
                    # alien_positions = move_aliens(alien_positions, ship_layout)  # Move aliens
                    # next_move = bot3_move(bot_position, captain_position, alien_positions, ship_layout)  # Bot replans and moves
                    # bot_position = next_move  
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
                        print("Successful Mission", success_count)
                        break
                    #visualize_layout(ship_layout,path, bot_position, captain_position, aliens)
                # else:
                #     #print("Mission Failed(3) : No Path Found")
                #     success = False  # Set success to False if the bot doesn't reach the captain within the specified steps
            else:
                #print("Mission Failed(4) : No Path Found")
                success = False  # Set success to False if there's no path from bot to captain

            survival = (alive and steps_taken == 1000) / (num_trials - success_count) if num_trials - success_count != 0 else 1.0


            results.append({
                'K': K,
                'Success_bot3': success,
                'Steps_bot3': steps_taken,
                'Survival_bot3': survival
            })
    # Print statement after each simulation
    print(f"Simulation for K={K} is complete")

    return pd.DataFrame(results)


def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D, grid):
    """Generate a random position within the grid bounds that is not blocked."""
    while True:
        x, y = random.randint(0, D-1), random.randint(0, D-1)
        if grid[x, y] == 1:  # 1 indicates an open cell
            return (x, y)
    #return random.randint(0, D-1), random.randint(0, D-1)

def get_neighbors_with_adjacent_blocked(position, grid, alien_positions):
    """Returns valid neighbors, excluding cells adjacent to aliens."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    adjacent_blocked = set()
    # Mark cells adjacent to aliens as blocked
    for alien in alien_positions:
        for dx, dy in directions:
            adj_x, adj_y = alien[0] + dx, alien[1] + dy
            if 0 <= adj_x < grid.shape[0] and 0 <= adj_y < grid.shape[1]:
                adjacent_blocked.add((adj_x, adj_y))
                
    neighbors = []
    for dx, dy in directions:
        nx, ny = position[0] + dx, position[1] + dy
        if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
            grid[nx, ny] == 1 and (nx, ny) not in alien_positions and 
            (nx, ny) not in adjacent_blocked):
            neighbors.append((nx, ny))
    return neighbors

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

# Should we remove this now ????
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
    # print(path)

# Find the shortest path from start to goal using A* pathfinding."""
def find_shortest_path(start, goal, grid, alien_positions, avoid_adjacent):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
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
            return path
        
        # Select neighbor function based on avoid_adjacent flag
        neighbors_func = get_neighbors_with_adjacent_blocked if avoid_adjacent else get_neighbors
        for neighbor in neighbors_func(current, grid, alien_positions):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == q_item[1] for q_item in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
    return []

def bot3_move(bot_position, captain_position, alien_positions, ship_layout):
    # First attempt: Avoid aliens and adjacent cells
    
    # start, goal, grid, alien_positions, avoid_adjacent
    path = find_shortest_path(bot_position, captain_position, ship_layout, alien_positions, True)
    # Reconstructs the path of the bot till the crew cell
    print("Path")
    print(path)
    if not path:
        # Fallback: Avoid only alien positions directly
        path = find_shortest_path(bot_position, captain_position, ship_layout, alien_positions, False)
    if path:
        next_step = path[1] # Why 1 , not 0?
    else:
        next_step = bot_position  # Stay in place if no path is found
    return next_step

def visualize_layout(layout, bot_position, captain_position, alien_positions, path):
    fig, ax = plt.subplots()
    ax.imshow(layout, cmap='binary', interpolation='nearest')  # Ship layout
    
    # Highlight dangerous adjacent cells
    danger_zone = set()
    for alien in alien_positions:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= alien[0] + dx < layout.shape[0] and 0 <= alien[1] + dy < layout.shape[1]:
                    danger_zone.add((alien[0] + dx, alien[1] + dy))
    
    for cell in danger_zone:
        ax.add_patch(plt.Circle((cell[1], cell[0]), 0.5, color='orange', alpha=0.3))  # Marking adjacent cells in orange

    # Highlight path
    if isinstance(path, list) and all(isinstance(p, tuple) and len(p) == 2 for p in path):
        for p in path:
            ax.add_patch(plt.Circle((p[1], p[0]), 0.3, color='yellow'))  # Correctly visualize path
            
    # Mark entities on the board
    ax.plot(bot_position[1], bot_position[0], 'bo')  # Bot in blue
    ax.plot(captain_position[1], captain_position[0], 'go')  # Captain in green
    for alien in alien_positions:
        ax.plot(alien[1], alien[0], 'rx')  # Aliens in red

    plt.xticks([]), plt.yticks([])
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

if __name__ == "__main__":
    #D = random.randint(1, 50)  # Dynamic ship size
    D = 30
    ship_layout = generate_ship_layout(D)
    print(ship_layout.shape)
    #print(ship_layout)
    #bot_position = random_position(D, ship_layout)
    #captain_position = random_position(D, ship_layout)

    #alien_count = random.randint(1, D//2)
    K_range = range(0,101,2)
    num_trials = 250

    bot3_data = {'K': [], 'Success_bot3': [], 'Steps_bot3': [], 'Survival_bot3': []}

    # Pass pre-generated ship layout, bot position, and captain position to the simulation method
    bot3_df = simulate_bot3(D, K_range, num_trials, ship_layout)
     # Print or visualize results as needed
    print(bot3_df.head())  # Example: Print the first few rows of the data
    #print(bot3_df.describe())  # Example: Print summary statistics of the data

    # Save data to CSV for later use
    bot3_df.to_csv('bot3_data.csv', index=False)

    # Visualizing the data
    plt.figure(figsize=(10, 6))
    #sns.lineplot(data=bot1_df, x='K', y='Success', estimator=np.mean)
    sns.lineplot(data=bot3_df, x='K', y='Success_bot3')
    plt.title('Bot 3 Success Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    #sns.lineplot(data=bot1_df, x='K', y='Steps', estimator=np.mean)
    sns.lineplot(data=bot3_df, x='K', y='Survival_bot3')
    plt.title('Bot 3 Survival Rate vs Number of Aliens')
    plt.xlabel('Number of Aliens (K)')
    plt.ylabel('Survival Rate')
    plt.legend()
    plt.show()
    # visualize_layout(ship_layout, bot_position, captain_position, aliens)

   