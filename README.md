# Roomba


https://rutgers.app.box.com/s/9p1h70l59sl7aqzvcrerw5xhgroyl1iy

Pathfinding Simulation in Ship Layout

Overview

This project simulates a pathfinding bot (Bot 1) navigating through a ship layout. The ship layout is generated with specified dimensions and obstacles, providing a maze for Bot 1 to solve. The goal is to find the shortest path from the bot's starting position to the captain's position, avoiding any obstacles and not considering the potential movements of aliens.

Features

Ship Layout Generation: Dynamically creates a grid-based ship layout with customizable dimensions and obstacle density.
Pathfinding Algorithm: Utilizes the A* algorithm for Bot 1 to find the shortest path to the captain.
Customizable Parameters: Allows for adjustments to the ship layout's size and complexity, as well as the pathfinding bot's behavior.

Files Description
ship_layout.py: Contains functions to generate the ship layout grid, including obstacle placement and dead-end processing.
bot1.py: Implements Bot 1's logic, including pathfinding using the A* algorithm and moving step-by-step towards the goal.