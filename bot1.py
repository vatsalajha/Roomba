# bot1.py
# Implements Bot 1: Plans the shortest path to the Captain, ignoring alien movements.

import numpy as np
import random
from queue import PriorityQueue
from ship_layout import generate_ship_layout

def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D):
    """Generate a random position within the grid bounds."""
    return random.randint(0, D-1), random.randint(0, D-1)

# Find the shortest path from start to goal using A* pathfinding."""
def find_shortest_path(start, goal, grid):
    open_set = PriorityQueue()
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
            return path[1:]  # Exclude the start position to get the next step
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
def bot1_move(bot_position, captain_position, ship_layout):
    path = find_shortest_path(bot_position, captain_position, ship_layout)
    # Move to the next cell in the path if it exists
    if path:
        next_step = path[0]
    else:
        next_step = bot_position  # Stay in place if no path is found
    return next_step

# Example usage placeholder (Actual logic to integrate with the simulation will be needed)
if __name__ == "__main__":
    # D = 10 yaha, I need to get this value from ship_layout (hardcoded for now)
    
    bot_position = random_position(10)
    captain_position = random_position(10) 
    ship_layout = generate_ship_layout(D)
    ship_layout = np.ones((10, 10), dtype=int)  # Example: Open grid
    ship_layout[5, :] = 0  # Example: Adding a row of blocked cells for complexity
    
    next_move = bot1_move(bot_position, captain_position, ship_layout)
    print(f"Bot 1 moves to: {next_move}")
