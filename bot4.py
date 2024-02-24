# Implement Bot 4: A bot with a custom strategy.

import numpy as np
from queue import PriorityQueue

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def calculate_risk_scores(ship_layout, aliens_positions, risk_range=2):
    D = ship_layout.shape[0]
    risk_scores = np.zeros_like(ship_layout, dtype=float)

    for alien in aliens_positions:
        for dx in range(-risk_range, risk_range+1):
            for dy in range(-risk_range, risk_range+1):
                nx, ny = alien[0] + dx, alien[1] + dy
                if 0 <= nx < D and 0 <= ny < D:
                    distance = max(abs(dx), abs(dy))
                    risk_increment = (risk_range + 1 - distance) / (risk_range + 1)
                    risk_scores[nx, ny] += risk_increment
    
    risk_scores = risk_scores / np.max(risk_scores)
    return risk_scores

def find_path_with_risk_assessment(start, goal, ship_layout, risk_scores):
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
            if 0 <= next[0] < D and 0 <= next[1] < D and ship_layout[next[0], next[1]] == 1:
                new_cost = cost_so_far[current] + 1 + risk_scores[next[0], next[1]]
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    frontier.put((priority, next))
                    came_from[next] = current

    path = []
    current = goal
    while current != start:
        if current is None:  # If no path found
            return []
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path

def bot4_move(bot_position, captain_position, ship_layout, aliens_positions):
    risk_scores = calculate_risk_scores(ship_layout, aliens_positions)
    path = find_path_with_risk_assessment(bot_position, captain_position, ship_layout, risk_scores)
    return path[0] if path else bot_position

# Example usage placeholder
if __name__ == "__main__":
    # Initialize ship_layout, bot_position, captain_position, and aliens_positions accordingly
    D = 10
    ship_layout = np.ones((D, D), dtype=int)
    ship_layout[2, 3:5] = 0  # Example obstacle
    ship_layout[6, 1:3] = 0  # Another example obstacle
    
    # Define the positions (row, column format)
    bot_position = (0, 0)  # Starting at the top-left corner
    captain_position = (9, 9)  # Captain is at the bottom-right corner
    aliens_positions = [(5, 5), (2, 6), (7, 8)]  # Placing aliens at different locations

    # Print the ship layout for visualization
    print("Ship Layout:")
    print(ship_layout)
    
    # Calculate the next move for Bot 4
    next_move = bot4_move(bot_position, captain_position, ship_layout, aliens_positions)
    print(f"Bot 4 moves to: {next_move}")
