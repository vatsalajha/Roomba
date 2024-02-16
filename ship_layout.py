# Grid Design

import numpy as np
import random

def initialize_ship_layout(D):
    # Create a DxD grid initialized with 0s (representing blocked cells)
    layout = np.zeros((D, D), dtype=int)
    return layout

def open_initial_cell(layout):
    D = layout.shape[0]
    # Choose a random cell (excluding the border to ensure it's an interior cell)
    x, y = random.randint(1, D-2), random.randint(1, D-2)
    layout[x, y] = 1  # Mark the cell as open

def get_neighbors(x, y, D):
    # Return a list of neighbor coordinates (up, down, left, right)
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    # Filter out neighbors that are outside the grid boundaries
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < D and 0 <= ny < D]
    return valid_neighbors

def iteratively_open_cells(layout):
    D = layout.shape[0]
    changed = True
    while changed:
        changed = False
        candidates = []
        for x in range(D):
            for y in range(D):
                if layout[x, y] == 0:  # If the cell is blocked
                    neighbors = get_neighbors(x, y, D)
                    open_neighbors = sum(layout[nx, ny] for nx, ny in neighbors)
                    if open_neighbors == 1:  # Exactly one open neighbor
                        candidates.append((x, y))
        if candidates:
            # Randomly select one and open it
            to_open = random.choice(candidates)
            layout[to_open[0], to_open[1]] = 1
            changed = True

def open_dead_end_neighbors(layout):
    D = layout.shape[0]
    dead_ends = []
    for x in range(D):
        for y in range(D):
            if layout[x, y] == 1:  # If the cell is open
                neighbors = get_neighbors(x, y, D)
                open_neighbors = sum(layout[nx, ny] for nx, ny in neighbors)
                if open_neighbors == 1:  # Exactly one open neighbor
                    dead_ends.append((x, y))
    # For about half of the dead ends, open a random blocked neighbor
    for dead_end in random.sample(dead_ends, len(dead_ends) // 2):
        neighbors = get_neighbors(dead_end[0], dead_end[1], D)
        blocked_neighbors = [(nx, ny) for nx, ny in neighbors if layout[nx, ny] == 0]
        if blocked_neighbors:
            to_open = random.choice(blocked_neighbors)
            layout[to_open[0], to_open[1]] = 1

def generate_ship_layout(D):
    layout = initialize_ship_layout(D)
    open_initial_cell(layout)
    iteratively_open_cells(layout)
    open_dead_end_neighbors(layout)
    return layout

# Example usage
D = 10  # Dimension of the grid
ship_layout = generate_ship_layout(D)
print(ship_layout)
