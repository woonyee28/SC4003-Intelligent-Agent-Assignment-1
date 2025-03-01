import math

def value_iteration(Grid):
    U = [[0 for _ in range(Grid.size)] for _ in range(Grid.size)]
    Ui = [[0 for _ in range(Grid.size)] for _ in range(Grid.size)]
    delta = math.inf
    epsilon = 0.05
    check = epsilon * (1-Grid.discount) / Grid.discount
    iteration = 0
    while (delta > check):
        for i in range(Grid.size):
            for j in range(Grid.size):
                U[i][j] = Ui[i][j]
        delta = 0
        for row in range(Grid.size):
            for column in range(Grid.size):
                if (row, column) in Grid.walls:
                    continue
                Ui[row][column] = Grid.get_reward(row, column) + Grid.value_get_expected_discount_utility(row, column, Ui)
                error = abs(Ui[row][column] - U[row][column])
                if error > delta:
                    delta = error
        iteration += 1
    print(f"Value Iteration converged after {iteration} iterations")
    return U

def value_extract_policy(Grid, U):
    policy = {}
    for row in range(Grid.size):
        for column in range(Grid.size):
            if (row, column) in Grid.walls:
                continue
            action_effects = {
                "UP": [("UP", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                "DOWN": [("DOWN", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                "LEFT": [("LEFT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)],
                "RIGHT": [("RIGHT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)]
            }
            old_state = Grid.cur_state
            Grid.cur_state = (row, column)
            best_action = None
            best_utility = float('-inf')
            for action in Grid.actions:
                utility = 0
                for effect_action, prob in action_effects[action]:
                    if Grid.check_boundary(effect_action):
                        Grid.move_agent(effect_action)
                        next_row, next_col = Grid.cur_state
                        reward = Grid.white_reward
                        if (next_row, next_col) in Grid.rewards:
                            reward = Grid.rewards[(next_row, next_col)]
                        utility += prob * (reward + Grid.discount * U[next_row][next_col])
                        Grid.cur_state = (row, column)
                    else:
                        reward = Grid.white_reward
                        if (row, column) in Grid.rewards:
                            reward = Grid.rewards[(row, column)]
                        utility += prob * (reward + Grid.discount * U[row][column])
                if utility > best_utility:
                    best_utility = utility
                    best_action = action
            Grid.cur_state = old_state
            policy[(row, column)] = best_action
    return policy

def take_optimal_action(Grid, policy):
    current_state = Grid.cur_state
    if current_state in policy:
        optimal_action = policy[current_state]
        Grid.move_agent(optimal_action)
        return optimal_action
    return None