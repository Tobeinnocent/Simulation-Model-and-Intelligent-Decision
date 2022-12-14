'''
This script is for creating a RL agent class object. This object has the 
following method:
    
    1) choose_action: this choose an action based on Q(s,a) and greedy eps
    2) learn: this updates the Q(s,a) table
    3) check_if_state_exist: this check if a state exist based on env feedback
'''

import numpy as np
import pandas as pd

class agent():
    
    def __init__(self, epsilon, lr, gamma, current_stock, debug, expected_stock, model_based):

        print("Created an Agent ...")
        self.actions = [-15,-10,-5,-1,0,1,5,10,15]
        self.reward = 0
        self.epsilon = epsilon #Exploitation的概率  epsilon-greedy
        self.lr = lr # learning rate
        self.gamma = gamma #折现率
        self.debug = debug
        self.current_stock = current_stock
        self.expected_stock = expected_stock
        self.model_based = model_based 

        # performance metric
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64) #状态动作价值表
        self.hourly_action_history = []
        self.hourly_stock_history = []

    def choose_action(self, s, ex):
        
        '''
        This funciton choose an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in subsequent hour (based on random forests prediction)
        
        Output:
            - action: number of bikes to move
        '''

        self.check_state_exist(s)
        self.current_stock = s
        self.expected_stock = ex
    
        if self.model_based == True:
            #Take an average of current stock and expected stock
            try:
                avg = int(round(0.5*s + 0.5*ex))
            except:
                avg = s
            self.check_state_exist(avg)
            valid_state_action = self.q_table.loc[avg, :]
        
        elif self.model_based == False:
            valid_state_action = self.q_table.loc[s, :]

        if np.random.uniform() < self.epsilon:#决定是Exploitation还是exploration
                        
            try:
                # find the action with the highest expected reward
                
                valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                action = valid_state_action.idxmax()
            
            except:
                # if action list is null, default to 0
                action = 0
                        
            if self.debug == True:
                print("Decided to Move: {}".format(action))
                        
        else:
            
            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
            except:
                action = 0
            
            if self.debug == True:
                print("Randomly Move: {}".format(action))

        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(s)
        
        return action

    def learn(self, s, a, r, s_, ex, g):

        
        '''
        This function updates Q tables after each interaction with the
        environment.
        Input: 
            - s: current bike stock
            - ex: expected bike stock in next hour
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
        Output: None
        '''
        
        if self.debug == True:
            print("Moved Bikes: {}".format(a))
            print("Old Bike Stock: {}".format(s))
            print("New Bike Stock: {}".format(s_))
            print("---")
        
        self.check_state_exist(s_)

        if self.model_based == False:
            q_predict = self.q_table.loc[s, a]
        elif self.model_based == True:
            avg = int(round(0.5*s + 0.5*ex))
            self.check_state_exist(avg)
            q_predict = self.q_table.loc[avg, a]
        

        if g == False:
            

            # Updated Q Target Value if it is not end of day  
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = r

        if self.model_based == False:
            self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        elif self.model_based == True:
            self.q_table.loc[avg, a] += self.lr * (q_target - q_predict)
        
        return

    def check_state_exist(self, state):
        # Add a new row with state value as index if not exist
        
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions), 
                        index = self.q_table.columns,
                        name = state
                        )
                )
            for a in self.q_table.columns:
                if state + a > 70:
                    self.q_table.loc[state,a] = -10-(state+a-70)
                elif state + a < 30:
                    self.q_table.loc[state,a] = -10-(30-state-a)
                else:
                    self.q_table.loc[state,a] = 20-abs(state+a-50)
        
        return

    def find_valid_action(self, state_action):
        
        '''
        This function check the validity acitons in a given state.
        Input: 
            - state_action: the current state under consideration
        Output:
            - state_action: a pandas Series with only the valid actions that
                            will not cause negative stock
        '''
        
        # remove action that will stock to be negative
        
        for action in self.actions:
            if self.current_stock + action < 0:
                
                if self.debug == True:
                    print("Drop action {}, current stock {}".format(action, self.current_stock))
                
                state_action.drop(index = action, inplace = True)
        
        return state_action

    def print_q_table(self):
            
            print(self.q_table)

    def get_q_table(self):
            
        return self.q_table
        
    def get_hourly_actions(self):
            
        return self.hourly_action_history
        
    def get_hourly_stocks(self):
            
        return self.hourly_stock_history

        
    def reset_hourly_history(self):
            
        self.hourly_action_history = []
        self.hourly_stock_history = []
