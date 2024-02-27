# bot1.py
# Implements Bot 1: Plans the shortest path to the Captain, ignoring alien movements.

import numpy as np
import random
from queue import PriorityQueue
from ship_layout import generate_ship_layout
import matplotlib.pyplot as plt

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

# Find the shortest path from start to goal using A* pathfinding."""
def find_shortest_path(start, goal, grid, alien_positions):
    open_set = PriorityQueue()
    # Storing the nodes to be explored
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current = open_set.get()[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path[0:]  
        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == q_item[1] for q_item in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
                    
    return []  # Return an empty path if there is no path to the goal
    #pass

def get_neighbors(position, grid):
    """Returns the valid neighbors for a position in the grid."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for direction in directions:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            if grid[neighbor] == 1:  # Ensure the neighbor is an open cell
                neighbors.append(neighbor)
    return neighbors

# Uses A* Algorithm to find shortest path to the captain
def bot1_move(bot_position, captain_position, ship_layout, alien_positions):
    # if not hasattr(bot1_move, "cached_path"):
    #     # Calculate the path with aliens considered only once
    #     bot1_move.cached_path = find_shortest_path(bot_position, captain_position, ship_layout, alien_positions)
    #     print("Path to be followed by bot", bot1_move.cached_path)

    # return bot1_move.cached_path
    #Consider aliens positiobs
    if not hasattr(bot1_move, "cached_path") or not bot1_move.cached_path:
        # Check for aliens' positions
        if alien_positions:
            # Calculate the path with aliens considered only once
            bot1_move.cached_path = find_shortest_path(bot_position, captain_position, ship_layout, alien_positions)
            print("Path to be followed by bot", bot1_move.cached_path)
        else:
            # Calculate the path without considering aliens
            bot1_move.cached_path = find_shortest_path(bot_position, captain_position, ship_layout)

    
        # Move to the next cell in the cached path if it exists
        # if bot1_move.cached_path and len(bot1_move.cached_path) > 1:
        #     next_step = bot1_move.cached_path.pop(0)
        # else:
        #     next_step = bot_position  # Stay in place if no path is found
        # return [next_step]
    return bot1_move.cached_path

def print_layout(layout, bot_position, captain_position, alien_positions, path):
    D = layout.shape[0]
    colors = {'0': 'black', '1': 'lightgrey', 'B': 'blue', 'A': 'red', 'C': 'green', 'P': 'orange'}

    # Create a plot
    plt.figure(figsize=(8, 8))

    # Plot each cell with the corresponding color
    for i in range(D):
        for j in range(D):
            color = colors[str(layout[i, j])]  # Convert numerical value to string for dictionary lookup
            #plt.scatter(j, D - i - 1, color=color, s=100, marker='s')  # Reversing the y-axis to match matrix indexing
            plt.fill([j, j+1, j+1, j], [D - i - 1, D - i - 1, D - i, D - i], color=color, edgecolor='black')  # Reversing the y-axis to match matrix indexing
            
    # Mark bot, captain, and aliens with their symbols
    plt.text(bot_position[1] + 0.5, D - bot_position[0] - 0.5, 'B', fontsize=12, color='blue', ha='center', va='center')
    plt.text(captain_position[1] + 0.5, D - captain_position[0] - 0.5, 'C', fontsize=12, color='green', ha='center', va='center')
    for alien_pos in alien_positions:
        plt.text(alien_pos[1] + 0.5, D - alien_pos[0] - 0.5, 'A', fontsize=12, color='red', ha='center', va='center')

    # Plot the entire path as 'P'
    for cell in path:
        plt.text(cell[1] + 0.5, D - cell[0] - 0.5, 'P', fontsize=12, color='orange', ha='center', va='center')
        
    # Customize plot appearance
    plt.xlim(0, D)
    plt.ylim(0, D)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix indexing
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, color='black', linewidth=2)

    # Show the plot
    plt.show()

# Example usage placeholder (Actual logic to integrate with the simulation will be needed)
if __name__ == "__main__":
    # D = 10 # yaha, I need to get this value from ship_layout (hardcoded for now)
    D = random.randint(3, 50)
    ship_layout = generate_ship_layout(D)
    print("Shape(size) of ship:", ship_layout.shape)
    print(ship_layout)
    bot_position = random_position(D, ship_layout)
    captain_position = random_position(D, ship_layout) 
    while captain_position == bot_position:  # Ensure bot and captain are not in the same position
        captain_position = random_position(D, ship_layout)

    alien_positions = [(3, 3), (7, 7)]  # Example alien positions

    path = bot1_move(bot_position, captain_position, ship_layout, alien_positions)
    print(f"Bot starts at: {bot_position}, Captain at: {captain_position}")

    for cell in path:
        print_layout(ship_layout, bot_position, captain_position, alien_positions, [cell])  # Pass each cell as a single-element list
        print(f"Bot moves to: {cell}")
        bot_position = cell

    # for cell in path:
    #     print_layout(ship_layout, bot_position, captain_position, alien_positions, path)
    #     print(f"Bot moves to: {cell}")
    #     bot_position = cell
    
    #ship_layout = np.ones((10, 10), dtype=int)  # Example: Open grid
    #ship_layout[5, :] = 0  # Example: Adding a row of blocked cells for complexity
    
    # while captain_position != bot_position:
    # next_move = bot1_move(bot_position, captain_position, ship_layout, alien_positions)
    # print(f"Bot starts at: {bot_position}, Captain at: {captain_position}")
    # print(f"Bot 1 moves to: {next_move}")
