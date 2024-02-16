import numpy as np
import random
from ship_layout import generate_ship_layout  # Assume this is your ship layout generator
# Import bots' move functions
from bot1 import bot1_move
# For simplicity, we'll only include bot1 in this example. Extend as needed.

def place_entities(ship_layout):
    open_cells = np.argwhere(ship_layout == 1)
    bot_pos, captain_pos, *alien_positions = random.sample(list(open_cells), k=K+2)
    return bot_pos, captain_pos, alien_positions

def move_aliens(alien_positions, ship_layout):
    new_positions = []
    for pos in alien_positions:
        x, y = pos
        possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_moves = [move for move in possible_moves if ship_layout[move] == 1]
        new_positions.append(random.choice(valid_moves) if valid_moves else pos)
    return new_positions

def check_collision(bot_pos, alien_positions):
    return bot_pos in alien_positions

def run_simulation(D, K):
    ship_layout = generate_ship_layout(D)
    bot_pos, captain_pos, alien_positions = place_entities(ship_layout)
    
    for step in range(1000):  # Maximum steps
        # Move bot (using bot1 as an example)
        bot_pos = bot1_move(bot_pos, captain_pos, ship_layout)
        
        # Check if bot has reached the Captain
        if np.array_equal(bot_pos, captain_pos):
            print("Bot has rescued the Captain!")
            return True
        
        # Move aliens
        alien_positions = move_aliens(alien_positions, ship_layout)
        
        # Check for collision
        if check_collision(bot_pos, alien_positions):
            print("Bot has been caught by an alien!")
            return False

    print("Simulation ended without rescuing the Captain.")
    return False

# Example parameters
D = 10  # Dimension of the ship layout
K = 3   # Number of aliens
run_simulation(D, K)
