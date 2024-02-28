# Implement Bot 4: A bot with a custom strategy.

import numpy as np
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue
from ship_layout import generate_ship_layout 

def heuristic(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D, ship_layout):
    """Generate a random position within the grid bounds that is not blocked."""
    while True:
        x, y = random.randint(0, D-1), random.randint(0, D-1)
        if ship_layout[x, y] == 1:  # Assuming 1 indicates an open cell
            return (x, y)

def visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, aliens_positions):
    """Visualize the ship layout with risk scores, bot, captain, and aliens."""
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis  # Use a colormap that represents risk well, like viridis
    risk_overlay = np.ma.masked_where(ship_layout == 0, risk_scores)  # Mask areas with no risk
    ax.imshow(ship_layout, cmap='Greys', interpolation='nearest')  # Ship layout in grey scale
    ax.imshow(risk_overlay, cmap=cmap, alpha=0.5, interpolation='nearest')  # Overlay risk scores

    # Mark bot, captain, and aliens
    ax.plot(bot_position[1], bot_position[0], 'bo', markersize=10)  # Bot in blue
    ax.plot(captain_position[1], captain_position[0], 'go', markersize=10)  # Captain in green
    for alien in aliens_positions:
        ax.plot(alien[1], alien[0], 'rx', markersize=10)  # Aliens in red with an 'x'

    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    plt.show()

def calculate_risk_scores(ship_layout, aliens_positions, risk_range=3):
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

def find_path_with_risk_assessment(start, goal, ship_layout, risk_scores, risk_multiplier=10):
    D = ship_layout.shape[0]
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            break

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < D and 0 <= next[1] < D and ship_layout[next[0], next[1]] != 0:  # Assume 0 as blocked
                new_cost = cost_so_far[current] + 1 + (risk_scores[next[0], next[1]] * risk_multiplier)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    frontier.put((priority, next))
                    came_from[next] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        if current is None:  # No path found
            return []
        path.append(current)
        current = came_from.get(current, None)
    path.reverse()  # Reverse to get path from start to goal
    return path

def place_aliens(D, grid, count, exclude_positions):
    aliens = []
    while len(aliens) < count:
        position = random_position(D, grid)
        if position not in exclude_positions:
            aliens.append(position)
            grid[position] = 2  # Assuming '2' marks an alien, adjust as needed
    return aliens


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
    path = find_path_with_risk_assessment(bot_position, captain_position, ship_layout, risk_scores)
    return path[0] if path else bot_position

# Example usage placeholder
if __name__ == "__main__":
    D = random.randint(1, 50)  # Dynamic ship size
    ship_layout = generate_ship_layout(D)
    print(ship_layout.shape)
    print(ship_layout)
    bot_position = random_position(D, ship_layout)
    captain_position = random_position(D, ship_layout)

    alien_count = random.randint(1, D//2)
    exclude_positions = [bot_position]
    #, captain_position]
    alien_positions = place_aliens(D, ship_layout, alien_count, exclude_positions)  # Reuse or define this function
    #print(alien_positions)
    while captain_position == bot_position:
        captain_position = random_position(D, ship_layout)

    path = find_path_with_risk_assessment(bot_position, captain_position, ship_layout, alien_positions, True)
    print(path)

    steps = 0
    max_steps = 1000 # To Prevent Infinite Loop
    while bot_position != captain_position and steps < max_steps:
        steps += 1
        alien_positions = move_aliens(alien_positions, ship_layout)  # Move aliens
        previous_position = bot_position
        risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
        next_move = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
        bot_position = next_move if next_move else bot_position  # Update bot position only if next_move is valid

        if bot_position == previous_position:
            print(f"Step {steps}: Bot is stuck at {bot_position}, trying to reach Captain at {captain_position}.")
        else:
            print(f"Step {steps}: Bot moved to {bot_position}, aiming for Captain at {captain_position}.")

        if bot_position == captain_position:
            print("Bot has successfully reached the captain!")
            break

    visualize_layout_with_risks(ship_layout, risk_scores, bot_position, captain_position, aliens_positions)
    # print(f"Next move for Bot 4: {next_move}")

