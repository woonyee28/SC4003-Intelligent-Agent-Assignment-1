from typing import List, Tuple, Dict

class Grid:
    def __init__(self, walls: List[Tuple[int,int]], start_state: Tuple[int,int], 
                 rewards: Dict[Tuple[int,int], float], intended_prob: float = 0.8, 
                 discount: float = 0.99, size: int = 6, white_reward: float = -0.05):
        """
        Initialize a Grid world for a Markov Decision Process.
        
        Parameters:
        walls (List[Tuple[int,int]]): Coordinates of wall cells that block movement
        start_state (Tuple[int,int]): Initial position of the agent
        rewards (Dict[Tuple[int,int], float]): Dictionary mapping positions to reward values
        intended_prob (float): Probability of moving in the intended direction (default: 0.8)
        discount (float): Discount factor for future rewards (default: 0.99)
        size (int): Size of the grid (default: 6x6)
        white_reward (float): Default reward for empty cells (default: -0.05)
        """
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
        """
        Check if an action is valid from the current state.
        
        Determines whether an action would move the agent to a valid cell
        (not a wall and within grid boundaries).
        
        Parameters:
        action (str): Direction to move ("UP", "DOWN", "LEFT", "RIGHT")
        
        Returns:
        bool: True if the move is valid, False otherwise
        """
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
        """
        Update the agent's position based on the given action.
        
        Moves the agent in the specified direction if the move is valid.
        If the move is invalid (wall or boundary), the agent stays in place.
        
        Parameters:
        action (str): Direction to move ("UP", "DOWN", "LEFT", "RIGHT")
        """
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
        """
        Get the reward value for a specific grid cell.
        
        Parameters:
        row (int): Row coordinate of the cell
        column (int): Column coordinate of the cell
        
        Returns:
        float: Reward value for the specified cell (0 for walls)
        """
        if self.world[row][column] == "W":
            return 0
        return self.world[row][column]

    def value_get_expected_discount_utility(self, row, column, Ui):
        """
        Calculate the maximum expected utility for a state in value iteration.
        
        Computes the Bellman update for a specific grid cell by considering all
        possible actions and their stochastic outcomes based on the transition model.
        
        Parameters:
        row (int): Row coordinate of the cell
        column (int): Column coordinate of the cell
        Ui (List[List[float]]): Current utility values for all grid cells
        
        Returns:
        float: Maximum expected utility across all possible actions
        """
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