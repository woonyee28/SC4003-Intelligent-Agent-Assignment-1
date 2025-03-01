import random

def policy_iteration(Grid):
    policy = {}
    for i in range(Grid.size):
        for j in range(Grid.size):
            if (i, j) not in Grid.walls:
                policy[(i, j)] = random.choice(Grid.actions)
    unchanged = False
    iteration = 0
    while not unchanged:
        U = policy_evaluation(Grid, policy)
        unchanged = True
        for i in range(Grid.size):
            for j in range(Grid.size):
                if (i, j) in Grid.walls:
                    continue
                old_state = Grid.cur_state
                Grid.cur_state = (i, j)
                best_action = None
                best_utility = float('-inf')
                for action in Grid.actions:
                    utility = 0
                    action_effects = {
                        "UP": [("UP", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                        "DOWN": [("DOWN", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                        "LEFT": [("LEFT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)],
                        "RIGHT": [("RIGHT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)]
                    }
                    for effect_action, prob in action_effects[action]:
                        if Grid.check_boundary(effect_action):
                            Grid.move_agent(effect_action)
                            next_row, next_col = Grid.cur_state
                            reward = Grid.white_reward
                            if (next_row, next_col) in Grid.rewards:
                                reward = Grid.rewards[(next_row, next_col)]
                            utility += prob * (reward + Grid.discount * U[next_row][next_col])
                            Grid.cur_state = (i, j)
                        else:
                            reward = Grid.white_reward
                            if (i, j) in Grid.rewards:
                                reward = Grid.rewards[(i, j)]
                            utility += prob * (reward + Grid.discount * U[i][j])
                    if utility > best_utility:
                        best_utility = utility
                        best_action = action
                Grid.cur_state = old_state
                if best_action != policy[(i, j)]:
                    policy[(i, j)] = best_action
                    unchanged = False
        iteration += 1
    print(f"Policy iteration converged after {iteration} iterations")
    return policy, U

def policy_evaluation(Grid, policy, max_iterations=100, theta=0.01):
    U = [[0 for _ in range(Grid.size)] for _ in range(Grid.size)]
    Ui = [[0 for _ in range(Grid.size)] for _ in range(Grid.size)]
    iteration = 0
    delta = float('inf')
    while delta > theta and iteration < max_iterations:
        for i in range(Grid.size):
            for j in range(Grid.size):
                U[i][j] = Ui[i][j]
        delta = 0
        for row in range(Grid.size):
            for column in range(Grid.size):
                if (row, column) in Grid.walls:
                    continue
                action = policy.get((row, column), Grid.actions[0])
                old_state = Grid.cur_state
                Grid.cur_state = (row, column)
                utility = 0
                action_effects = {
                    "UP": [("UP", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                    "DOWN": [("DOWN", Grid.intended_prob), ("LEFT", (1-Grid.intended_prob)/2), ("RIGHT", (1-Grid.intended_prob)/2)],
                    "LEFT": [("LEFT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)],
                    "RIGHT": [("RIGHT", Grid.intended_prob), ("UP", (1-Grid.intended_prob)/2), ("DOWN", (1-Grid.intended_prob)/2)]
                }
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
                Grid.cur_state = old_state
                Ui[row][column] = Grid.get_reward(row, column) + utility
                error = abs(Ui[row][column] - U[row][column])
                if error > delta:
                    delta = error
        iteration += 1
    return Ui