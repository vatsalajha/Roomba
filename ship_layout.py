# Grid Design

import numpy as np
import random
import matplotlib.pyplot as plt

def initialize_ship_layout(D):
    # Create a DxD grid initialized with 0s (representing blocked cells)
    layout = np.zeros((D, D), dtype=int)
    return layout

def open_initial_cell(layout):
    D = layout.shape[0]
    # Choose a random cell (excluding the border to ensure it's an interior cell)
    x, y = random.randint(1, D-2), random.randint(1, D-2)
    layout[x, y] = 1  # Mark the cell as open
    print(x, y)

def get_neighbors(x, y, D):
    # Return a list of neighbor coordinates (up, down, left, right)
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    # Filter out neighbors that are outside the grid boundaries
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < D and 0 <= ny < D]
    #print(valid_neighbors)
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
                        candidates.append((x, y)) # Open neighbours max 1 hone chahiye isliye
        if candidates:
            # Randomly select one and open it
            to_open = random.choice(candidates)
            layout[to_open[0], to_open[1]] = 1
            changed = True

# Opening half of the dead ends ka blocked neighbours cells
def open_dead_end_neighbors(layout, fraction=0.5):
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
    for dead_end in random.sample(dead_ends, int(len(dead_ends) * fraction)):
        neighbors = get_neighbors(dead_end[0], dead_end[1], D)
        blocked_neighbors = [(nx, ny) for nx, ny in neighbors if layout[nx, ny] == 0]
        if blocked_neighbors:
            to_open = random.choice(blocked_neighbors)
            layout[to_open[0], to_open[1]] = 1

def visualize_layout(layout):
    plt.imshow(layout, cmap='Greys', interpolation='nearest')
    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    plt.show()

def generate_ship_layout(D, dead_end_opening_fraction=0.5):
    layout = initialize_ship_layout(D)
    open_initial_cell(layout)
    iteratively_open_cells(layout)
    open_dead_end_neighbors(layout, fraction=dead_end_opening_fraction)
    return layout

# Example usage
D = 10  # Dimension of the grid
dead_end_opening_fraction = 0.5
ship_layout = generate_ship_layout(D, dead_end_opening_fraction)

print(ship_layout)
visualize_layout(ship_layout)