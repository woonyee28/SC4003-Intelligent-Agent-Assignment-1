from typing import List, Tuple, Dict

class Grid:
    def __init__(self, walls: List[Tuple[int,int]], start_state: Tuple[int,int], 
                 rewards: Dict[Tuple[int,int], float], intended_prob: float = 0.8, 
                 discount: float = 0.99, size: int = 6, white_reward: float = -0.05):
        self.size = size
        self.world = [[0 for _ in range(size)] for _ in range(size)] # states
        self.walls = walls
        self.cur_state = start_state
        self.rewards = rewards # reward R(s)
        self.intended_prob = intended_prob # transition model for intended outcome
        self.discount = discount # γ discount
        self.white_reward = white_reward
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"] # actions
        for wall in walls:
            i, j = wall
            self.world[i][j] = "W"

        for pos, value in rewards.items():
            i, j = pos
            self.world[i][j] = value

    def check_boundary(self, action):
        # True == OK, False == No go
        i, j = self.cur_state
        match action:
            case "DOWN":
                if i+1 < self.size and self.world[i+1][j] != "W":
                    return True
            case "UP":
                if i-1 >= 0 and self.world[i-1][j] != "W":
                    return True
            case "LEFT":
                if j-1 >= 0 and self.world[i][j-1] != "W":
                    return True
            case "RIGHT":
                if j+1 < self.size and self.world[i][j+1] != "W":
                    return True
        return False

    def move_agent(self, action):
        i, j = self.cur_state
        if self.check_boundary(action):
            match action:
                case "DOWN":
                    self.cur_state = (i+1, j)
                case "UP":
                    self.cur_state = (i-1, j)
                case "LEFT":
                    self.cur_state = (i, j-1)
                case "RIGHT":
                    self.cur_state = (i, j+1)

    def get_reward(self, row, column):
        if self.world[row][column] == "W":
            return 0
        return self.world[row][column]

    def value_get_expected_discount_utility(self, row, column, Ui):
        action_effects = {
            "UP": [("UP", self.intended_prob), ("LEFT", (1-self.intended_prob)/2), ("RIGHT", (1-self.intended_prob)/2)],
            "DOWN": [("DOWN", self.intended_prob), ("LEFT", (1-self.intended_prob)/2), ("RIGHT", (1-self.intended_prob)/2)],
            "LEFT": [("LEFT", self.intended_prob), ("UP", (1-self.intended_prob)/2), ("DOWN", (1-self.intended_prob)/2)],
            "RIGHT": [("RIGHT", self.intended_prob), ("UP", (1-self.intended_prob)/2), ("DOWN", (1-self.intended_prob)/2)]
        }
        action_utilities = {}
        for action in self.actions:
            utility = 0
            old_state = self.cur_state
            self.cur_state = (row, column)
            for effect_action, prob in action_effects[action]:
                if self.check_boundary(effect_action):
                    self.move_agent(effect_action)
                    next_row, next_col = self.cur_state
                    reward = self.white_reward
                    if (next_row, next_col) in self.rewards:
                        reward = self.rewards[(next_row, next_col)]
                    utility += prob * (reward + self.discount * Ui[next_row][next_col])
                    self.cur_state = (row, column)
                else:
                    reward = self.white_reward
                    if (row, column) in self.rewards:
                        reward = self.rewards[(row, column)]
                    utility += prob * (reward + self.discount * Ui[row][column])
            self.cur_state = old_state
            action_utilities[action] = utility
        return max(action_utilities.values()) if action_utilities else self.white_reward