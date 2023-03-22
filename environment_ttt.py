"""
A representation of a tictactoe environment that users can play in.
The agent's turn is determined as we initially set it up.

An example of environment implementation.
"""

EMPTY   = E = 0
CIRCLE  = O = 1
CROSS   = X = 2
DRAW    = 3
ERROR   = 4
ONGOING = 999

ACTION_SPACE_SIZE = 9
OBSERVATION_SPACE_SIZE = 3**9

REWARD_FOR_WIN = 1
REWARD_FOR_LOSS = -1
REWARD_FOR_DRAW = 0.75
REWARD_FOR_ERROR = -5
REWARD_FOR_ONGOING = 0


class Tictactoe:
    
    # Defines a new tic tac toe environment, with the given rewards for the different 
    # game states: 1) ai wins, 2) ai loses, 3) ai draws, 4) ai plays a square already filled,
    #              5) game is still ongoing.
    # The ai turn is given to determine rewards.
    def __init__(self, r_win=REWARD_FOR_WIN, r_loss=REWARD_FOR_LOSS, 
                 r_draw=REWARD_FOR_DRAW, r_error=REWARD_FOR_ERROR, 
                 r_ongoing=REWARD_FOR_ONGOING, ai_turn=CIRCLE):
        
        self.end_condition = ONGOING # can be circle(1), cross(2), draw(3), 
                                     #  error(4) or 999 for ongoing game.
        self.state = [E]*9
        self.ai_turn = ai_turn
        #reward values
        self.reward_for_win   = r_win
        self.reward_for_loss  = r_loss
        self.reward_for_draw  = r_draw
        self.reward_for_error = r_error
        self.reward_when_not_done = r_ongoing
    
    # Resets the environment to an empty grid of tic tac toe.
    def reset(self):
        self.end_condition = ONGOING
        self.state = [E]*9
        return self.state
    
    # Executes a single step in the environment with 
    def step(self, state, action):
        def _chosen_square_is_empty():
            return (state[action] == EMPTY)
        
        def _update_state():
            new_state = []
            for i in state:
                new_state.append(i) 
            new_state[action] = Tictactoe.get_turn(state) 
            return new_state
        
        def _get_end_condition(s):
            _, end_condition = Tictactoe.is_terminal(s)
            return end_condition
        
        def _get_reward(end_condition):
            opponent_turn = CROSS if self.ai_turn is CIRCLE else CIRCLE
            
            if end_condition is self.ai_turn:
                return self.reward_for_win
            elif end_condition is opponent_turn:
                return self.reward_for_loss
            elif end_condition is DRAW:
                return self.reward_for_draw
            elif end_condition is ONGOING:
                return self.reward_when_not_done
            
        if _chosen_square_is_empty():
            self.state = _update_state()
            self.end_condition = _get_end_condition(self.state)
            reward = _get_reward(self.end_condition)
            done = (end_condition != ONGOING)
            return self.state, reward, done, False, ""
            #observation; reward; terminated; truncated; info.
        else:
            end_condition = ERROR
            return state, self.reward_for_error, False, True, ""
            #observation; reward; terminated; truncated; info.

    def render(self):
        def _convert_number_to_mark(n):
            if n == EMPTY:
                return " "
            elif n == CIRCLE:
                return "o"
            elif n == CROSS:
                return "x"
            
        Tictactoe.print_board(list(map(_convert_number_to_mark, self.state)))
    
    def set_state(self, state):
        self.reset()
        self.state = state

    def get_random_action(self, state):
        return


    #Other functions useful on tic tac toe board states
    @staticmethod
    def get_empty_squares(state):
        result = []
        for i in range(9):
            if state[i] is EMPTY:
                result.append(i)
        return result

    @staticmethod
    def get_turn(state):
        if len(Tictactoe.get_empty_squares(state)) % 2 == 0:
            return CROSS
        else:
            return CIRCLE

    @staticmethod    
    def print_board(b):
        print("_______")
        print("|" + b[0] + "|" + b[1] + "|" + b[2] + "|")
        print("|" + b[3] + "|" + b[4] + "|" + b[5] + "|")
        print("|" + b[6] + "|" + b[7] + "|" + b[8] + "|")
        print("-------")

    @staticmethod
    def is_terminal(state):
        def _at_least_one_line_is_full_with(s, mark):
            lines = ((0,1,2), (3,4,5), (6,7,8),
                     (0,3,6), (1,4,7), (2,5,8),
                     (0,4,8), (2,4,6))
            
            for L in lines:
                first, second, third = L[0], L[1], L[2]
                if s[first] == s[second] == s[third] == mark:
                    return True
    
            return False
        
        is_terminal = False
        
        if _at_least_one_line_is_full_with(state, CIRCLE):
            end_condition = CIRCLE
            is_terminal = True
        elif _at_least_one_line_is_full_with(state, CROSS):
            end_condition = CROSS
            is_terminal = True
        elif EMPTY not in state:
            end_condition = DRAW
            is_terminal = True
        else:
            end_condition = ONGOING
        
        return is_terminal, end_condition
