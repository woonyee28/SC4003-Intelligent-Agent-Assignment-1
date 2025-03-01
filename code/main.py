from grid import Grid
from value_iteration import value_iteration, value_extract_policy
from policy_iteration import policy_iteration
import curses

def take_optimal_action(grid, policy):
    current_state = grid.cur_state
    if current_state in policy:
        optimal_action = policy[current_state]
        grid.move_agent(optimal_action)
        return optimal_action
    return None

def main(stdscr):
    curses.curs_set(0)
    map1 = {
        "walls": [(0, 1), (1, 4), (4, 1), (4, 2), (4, 3)],
        "rewards": {
            (0, 0): 1, (0, 2): 1, (0, 5): 1, 
            (1, 1): -1, (1, 3): 1, (1, 5): -1,          
            (2, 2): -1, (2, 4): 1,                  
            (3, 3): -1, (3, 5): 1,           
            (4, 4): -1
        },
        "start_state": (3, 2),
        "size": 6
    }
    map2 = {
        "walls": [
            # Vertical wall sections
            *[(x, y) for x in range(3, 5) for y in [2, 7, 12]],
            *[(x, y) for x in range(8, 10) for y in [0, 5, 10, 14]],
            *[(x, y) for x in range(12, 14) for y in [3, 8, 13]],
            
            # Horizontal wall sections
            *[(x, y) for y in range(6, 8) for x in [1, 6, 11, 14]],
            *[(x, y) for y in range(11, 13) for x in [0, 5, 10]],
            *[(x, y) for y in range(3, 5) for x in [7, 12]]
        ],
        "rewards": {
            # Positive rewards
            (0, 0): 1, (0, 14): 1, 
            (5, 5): 1, (5, 10): 1,
            (7, 7): 1, (7, 0): 1,
            (10, 14): 1, (10, 5): 1,
            (14, 0): 1, (14, 14): 1,
            
            # Negative rewards
            (2, 2): -1, (2, 12): -1,
            (6, 6): -1, (6, 8): -1,
            (9, 9): -1, (9, 4): -1,
            (13, 13): -1, (13, 1): -1
        },
        "start_state": (7, 7),  # Near center of the map
        "size": 20
    }
    
    # map2 = {
    #     "walls": [
    #         (0, 2), (0, 3), (1, 0), (1, 2), (1, 5),
    #         (2, 2), (2, 4), (3, 0), (3, 4),
    #         (4, 2), (4, 3), (5, 3), (5, 5),
    #         (2, 7), (2, 9), (7, 0), (8, 4),
    #         (4, 8), (4, 7), (7, 3), (9, 5),
    #         (9, 9), (8, 7), (7, 9)
    #     ],
    #     "rewards": {
    #         (0, 0): -1, (0, 5): 1, 
    #         (2, 1): -1, (2, 3): 1, 
    #         (3, 2): -1, (3, 5): 1,  
    #         (5, 0): 1,  (5, 4): -1, 
    #         (4, 1): -1, (1, 4): 1,
    #         (0, 8): 1,  (0, 9): -1,
    #         (1, 7): -1, (1, 9): 1,
    #         (3, 7): 1,  (3, 9): -1,
    #         (5, 8): -1, (5, 9): 1,
    #         (6, 2): 1,  (6, 5): -1,
    #         (7, 5): -1, (7, 7): 1,
    #         (8, 0): 1,  (8, 8): -1,
    #         (9, 2): -1, (9, 7): 1
    #     },
    #     "start_state": (5, 1),
    #     "size": 10
    # }
    current_map = map1
    current_algorithm = "value"
    while True:
        stdscr.clear()
        stdscr.addstr(1, 10, "SC4003 Assignment 1: Agent Decision Making")
        stdscr.addstr(2, 20, "Author: Woon Yee")
        stdscr.addstr(4, 5, "1. Map 1 with Value Iteration")
        stdscr.addstr(5, 5, "2. Map 1 with Policy Iteration")
        stdscr.addstr(6, 5, "3. Map 2 with Value Iteration")
        stdscr.addstr(7, 5, "4. Map 2 with Policy Iteration")
        stdscr.addstr(8, 5, "5. Quit")
        stdscr.addstr(10, 5, "Enter your choice: ")
        stdscr.refresh()
        
        choice = stdscr.getch()
        temp = 0
        
        if choice == ord('1'):
            current_map = map1
            current_algorithm = "value"
            temp = 1
        elif choice == ord('2'):
            current_map = map1
            current_algorithm = "policy"
            temp = 1
        elif choice == ord('3'):
            current_map = map2
            current_algorithm = "value"
            temp = 2
        elif choice == ord('4'):
            current_map = map2
            current_algorithm = "policy"
            temp = 2
        elif choice == ord('5'):
            break
        else:
            continue
        print("Map chosen: " + str(temp))
        grid = create_grid(current_map)
        utilities, policy = run_algorithm(current_algorithm, grid)
        visualize_grid_and_handle_input(stdscr, grid, utilities, policy, current_algorithm)

def create_grid(map_data):
    return Grid(
        walls=map_data["walls"],
        start_state=map_data["start_state"],
        rewards=map_data["rewards"],
        intended_prob=0.8,
        discount=0.99,
        size=map_data["size"]
    )

def run_algorithm(algorithm, grid):
    if algorithm == "value":
        utilities = value_iteration(grid)
        policy = value_extract_policy(grid, utilities)
        return utilities, policy
    else:
        policy, utilities = policy_iteration(grid)
        return utilities, policy

def visualize_grid_and_handle_input(stdscr, grid, utilities, policy, algorithm):
    visualize_grid(stdscr, grid, utilities, policy, algorithm)
    key = None
    while key != ord('q'):
        key = stdscr.getch()
        if key == ord('o'):
            take_optimal_action(grid, policy)
        elif key == curses.KEY_UP:
            if grid.cur_state[0] > 0 and (grid.cur_state[0]-1, grid.cur_state[1]) not in grid.walls:
                grid.cur_state = (grid.cur_state[0]-1, grid.cur_state[1])
        elif key == curses.KEY_DOWN:
            if grid.cur_state[0] < grid.size-1 and (grid.cur_state[0]+1, grid.cur_state[1]) not in grid.walls:
                grid.cur_state = (grid.cur_state[0]+1, grid.cur_state[1])
        elif key == curses.KEY_LEFT:
            if grid.cur_state[1] > 0 and (grid.cur_state[0], grid.cur_state[1]-1) not in grid.walls:
                grid.cur_state = (grid.cur_state[0], grid.cur_state[1]-1)
        elif key == curses.KEY_RIGHT:
            if grid.cur_state[1] < grid.size-1 and (grid.cur_state[0], grid.cur_state[1]+1) not in grid.walls:
                grid.cur_state = (grid.cur_state[0], grid.cur_state[1]+1)
        elif key == ord('b'):  
            return
            
        visualize_grid(stdscr, grid, utilities, policy, algorithm)

def visualize_grid(stdscr, grid, utilities, policy, algorithm):
    stdscr.clear()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_RED)    
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN) 
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)  
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    
    start_y, start_x = 4, 5
    stdscr.addstr("      SC4003 Assignment 1: Agent Decision Making")
    stdscr.addstr("\n")
    stdscr.addstr("                 Author: Woon Yee               ")
    algo_name = "Value Iteration" if algorithm == "value" else "Policy Iteration"
    stdscr.addstr(2, 12, f"Algorithm: {algo_name}")
    
    action_symbols = {
        "UP": "↑",
        "DOWN": "↓",
        "LEFT": "←",
        "RIGHT": "→"
    }
    for j in range(grid.size):
        stdscr.addstr(start_y - 1, start_x + 4 * j + 2, str(j))
    
    for i in range(grid.size):
        stdscr.addstr(start_y + 2 * i, start_x - 2, str(i))
        for j in range(grid.size):
            cell_y = start_y + 2 * i
            cell_x = start_x + 4 * j
            stdscr.addstr(cell_y, cell_x, "+---+")
            stdscr.addstr(cell_y + 1, cell_x, "|   |")
            stdscr.addstr(cell_y + 2, cell_x, "+---+")
            cell = grid.world[i][j]
            
            if (i, j) == grid.cur_state:
                color = curses.color_pair(4)
                content =  " A "
            elif cell == "W":
                color = curses.color_pair(5)
                content = "###"
            elif cell == 1:
                color = curses.color_pair(3)
                content = f"+{cell} "
            elif cell == -1:
                color = curses.color_pair(2)
                content = f"{cell} "
            else:
                if (i, j) in policy:
                    color = curses.color_pair(6)
                    content = f" {action_symbols[policy[(i, j)]]} "
                else:
                    color = curses.color_pair(1)
                    content = "   "
            
            stdscr.addstr(cell_y + 1, cell_x + 1, content, color)
    
    legend_y = start_y
    legend_x = start_x + 4 * grid.size + 6
    stdscr.addstr(legend_y, legend_x, "Legend:")
    stdscr.addstr(legend_y + 1, legend_x, "A", curses.color_pair(4))
    stdscr.addstr(legend_y + 1, legend_x + 2, "- Agent Position")
    stdscr.addstr(legend_y + 2, legend_x, "###", curses.color_pair(5))
    stdscr.addstr(legend_y + 2, legend_x + 4, "- Wall")
    stdscr.addstr(legend_y + 3, legend_x, "+1", curses.color_pair(3))
    stdscr.addstr(legend_y + 3, legend_x + 3, "- Positive Reward")
    stdscr.addstr(legend_y + 4, legend_x, "-1", curses.color_pair(2))
    stdscr.addstr(legend_y + 4, legend_x + 3, "- Negative Reward")
    stdscr.addstr(legend_y + 5, legend_x, " ↑ ", curses.color_pair(6))
    stdscr.addstr(legend_y + 5, legend_x + 4, "- Optimal Policy")
    
    stdscr.addstr(legend_y + 7, legend_x, "Press 'q' to quit")
    stdscr.addstr(legend_y + 8, legend_x, "Press 'o' for optimal move")
    stdscr.addstr(legend_y + 9, legend_x, "Press 'b' to return to menu")
    stdscr.addstr(legend_y + 10, legend_x, "Use arrow keys to move manually")
    
    current_state = grid.cur_state
    current_utility = utilities[current_state[0]][current_state[1]]
    current_policy = policy.get(current_state, "None")
    
    stdscr.addstr(legend_y + 12, legend_x, f"Current state: {current_state}")
    stdscr.addstr(legend_y + 13, legend_x, f"Utility: {current_utility:.4f}")
    stdscr.addstr(legend_y + 14, legend_x, f"Policy: {current_policy}")
    
    stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(main)