import numpy as np

np.random.seed(62)
np.set_printoptions(threshold=np.inf)

"""
This big function creates a transitional matrix which has the information of
From state i, with an action k, probability of moving to jth state.
printing transitional_matrix[:,:,i] where i is 0-1-2-3(action indexes)
will show a clearer image of what this does.
transitional_matrix[:,:,0] = [[0.9 0.1 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.1 0.8 0.1 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.1 0.8 0.1 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.8 0.  0.  0.  0.2 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.8 0.  0.  0.  0.1 0.1 0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.8 0.  0.  0.1 0.1 0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.8 0.  0.  0.  0.1 0.1 0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.1 0.8 0.1 0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.1 0.  0.1 0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.1 0.1 0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.1 0.1 0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                              [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]
For example from s0, with action north, we have a 0.9 probability of staying in s0 and 0.1 probability of staying in s1.
"""
def create_transitional_matrix(state_number, action_number):
    transitional_model = np.zeros((state_number,state_number,action_number))
    for i in range(state_number):#state
        for j in range(state_number):#next_state
            for k in range(action_number):#action 0 = n, 1 = w, 2 = s, 3 = e
                if i == 0:
                    if j == 0:
                        if k == 0:
                            transitional_model[i,j,k] = p + p1
                        elif k == 1:
                            transitional_model[i,j,k] = p + p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    elif j == 1:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 4:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 1:
                    if j == 0:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 1:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1 + p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1 + p1
                    elif j == 2:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                elif i == 2:
                    if j == 1:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 2:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    elif j == 3:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 6:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 4:
                    if j == 0:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    elif j == 4:
                        if k == 0:
                            transitional_model[i,j,k] = p1 + p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1 + p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 8:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 6:
                    if j == 2:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 6:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 7:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 10:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 7:
                    if j == 3:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 6:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 7:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 11:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 8:
                    if j == 4:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 8:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 9:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 12:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 9:
                    if j == 8:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 9:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    elif j == 10:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 13:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 10:
                    if j == 6:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 9:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 11:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 14:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 11:
                    if j == 7:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 10:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                    elif j == 11:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
                    elif j == 15:
                        if k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                elif i == 12:
                    if j == 8:
                        if k == 0:
                            transitional_model[i,j,k] = p
                        elif k == 1:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    if j == 12:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 1:
                            transitional_model[i,j,k] = p + p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1 + p
                        elif k == 3:
                            transitional_model[i,j,k] = p1
                    elif j == 13:
                        if k == 0:
                            transitional_model[i,j,k] = p1
                        elif k == 2:
                            transitional_model[i,j,k] = p1
                        elif k == 3:
                            transitional_model[i,j,k] = p
               
    return transitional_model         

"""Finds and returns the index of the best action available at a particular state(this state is defined in the state array where
 the currently checked state's index's value is 1, where as others are 0) by iterating over the all actions 
using the transitional model and utility_array

"""
def find_the_best_action_for_a_state(utility_array, state_array, transitional_model):
    action_array = np.array([0.0,0.0,0.0,0.0])#north-east-south-west
    for i in range(4):
        action_array[i] = 1 - 0.01 + np.sum(np.multiply(utility_array, np.dot(state_array, transitional_model[:,:,i])))
    return action_array.argmax()  

"""
Creates a basic reward matrix with the given values from the environment. It also assigns the other states' rewards with the 
specified "state_reward" parameter
Returns the reward matrix
"""
def create_reward_matrix(state_reward):
    reward_matrix = np.array([0.0,0.0,0.0,1.0,
                              0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,
                              0.0,1.0,-10.0,10.0])  
    for i in range(16):
        if i not in [3,5,13,14,15]:
            reward_matrix[i] = state_reward
    return reward_matrix

"""
Finds the utility matrix for the value iteration algorithm.
First it creates an array of zeros with a size of 16
Then it loops over all states for the iteration count, finding the current state's reward, 
creating a state_value_matrix which will help us find that state's utility value using the transitional_matrix, utility_matrix, 
reward and the discount_factor
Then the found value is updated in the utility_matrix. 
This operation is done for iteration_count * 16(state count) times
At the end, found utility matrix is returned.
"""     
def find_utility_matrix_for_value_iteration(iteration_count, state_reward_matrix, transitional_matrix, discount_factor):
    utility_matrix = np.zeros(16)
    for i in range(iteration_count):
        for j in range(16):#number of states
            reward = state_reward_matrix[j]
            state_value_matrix = np.zeros((1, 16))
            state_value_matrix[0,j] = 1.0
            utility_matrix[j] = find_state_utility(state_value_matrix, transitional_matrix, utility_matrix, reward, discount_factor)
    return utility_matrix

"""
Basic printing function that prints the utility matrix in a 4x4 manner.
"""
def print_utility_matrix(utility_matrix):
    print(utility_matrix[0:4])
    print(utility_matrix[4:8])
    print(utility_matrix[8:12])
    print(utility_matrix[12:16])

"""
From this line to line 524, all the functions are used for creating a policy string such as ">>v1vX>v>>>v^1-1010" for a given utility array.
These functions also use the current policy string to make these updates permanent and the state index to find the neighbouring state values
to determine which neighbouring state has the better utility value.
At the end policy string for the given utility array is returned.
"""
def update_policy_string_s0(utility_array, policy_string, state_index):
    current = utility_array[state_index]
    east = utility_array[state_index+1]
    south = utility_array[state_index+4]
    best = max(current, east, south)
    if best == east:
        policy_string += ">"
    elif best == south:
        policy_string += "v"
    else:
        if east > south:
            policy_string += "^"
        else:
            policy_string += "<"
    return policy_string

def update_policy_string_s1_s2(utility_array, policy_string, state_index):
    current = utility_array[state_index]
    east = utility_array[state_index+1]
    south = utility_array[state_index+4]
    west = utility_array[state_index-1]
    best = max(current, east, south, west)
    if best == east:
        policy_string += ">"
    elif best == south:
        policy_string += "v"
    elif best == west:
        policy_string += "<"
    else:
        policy_string += "^"
    return policy_string

def update_policy_string_s4_s8(utility_array, policy_string, state_index):
    current = utility_array[state_index]
    east = utility_array[state_index+1]
    south = utility_array[state_index+4]
    north = utility_array[state_index-4]
    best = max(current, east, south, north)
    if best == east:
        policy_string += ">"
    elif best == south:
        policy_string += "v"
    elif best == north:
        policy_string += "^"
    else:
        policy_string += "<"
    return policy_string

def update_policy_string_s6_s9_s10(utility_array, policy_string, state_index):
    east = utility_array[state_index+1]
    south = utility_array[state_index+4]
    north = utility_array[state_index-4]
    west = utility_array[state_index-1]
    best = max(east, south, north, west)
    if best == east:
        policy_string += ">"
    elif best == south:
        policy_string += "v"
    elif best == north:
        policy_string += "^"
    else:
        policy_string += "<"
    return policy_string

def update_policy_string_s7_s11(utility_array, policy_string, state_index):
    current = utility_array[state_index]
    south = utility_array[state_index+4]
    north = utility_array[state_index-4]
    west = utility_array[state_index-1]
    best = max(current, south, north, west)
    if best == west:
        policy_string += ">"
    elif best == south:
        policy_string += "v"
    elif best == north:
        policy_string += "^"
    else:
        policy_string += ">"
    return policy_string

def update_policy_string_s12(utility_array, policy_string, state_index):
    current = utility_array[state_index]
    north = utility_array[state_index-4]
    east = utility_array[state_index+1]
    best = max(current, east, north)
    if best == north:
        policy_string += "^"
    elif best == east:
        policy_string += ">"
    else:
        if north > east:
            policy_string += "<"
        else:
            policy_string += "v"
    return policy_string

def create_policy_string_from_utility_array(utility_array):
    policy_string = ""
    for i in range(16):
        if i == 0:
            policy_string = update_policy_string_s0(utility_array, policy_string, i)
        elif i == 1 or i == 2:
            policy_string = update_policy_string_s1_s2(utility_array, policy_string, i)
        elif i == 4 or i == 8:
            policy_string = update_policy_string_s4_s8(utility_array, policy_string, i)
        elif i == 6 or i == 9 or i == 10:
            policy_string = update_policy_string_s6_s9_s10(utility_array, policy_string, i)
        elif i == 7 or i == 11:
            policy_string = update_policy_string_s7_s11(utility_array, policy_string, i)
        elif i == 12:
            policy_string = update_policy_string_s12(utility_array, policy_string, i)
        elif i == 3 or i == 13:
            policy_string += "1"
        elif i == 5:
            policy_string += "X"
        elif i == 14:
            policy_string += "-10"
        elif i == 15:
            policy_string += "10"
    return policy_string


"""
This is the main policy iteration function.
It first defines the utility_matrix as an array of zeros for the size of 16
Then it updates the utility matrix to be the same as the reward matrix(initial reward matrix = [0,0,0,1
                                                                                                0,0,0,0
                                                                                                0,0,0,0
                                                                                                0,1,-10,10])
I used a loop to do this operation because there was some mutable list errors with other implementation ways. I needed a deep copy.

Then I created a count to know how many iterations it took to converge or basically to see which iteration count resulted in which utility array

Now the main policy iteration starts. I created an infinite loop and created a u1 array which is the same array as the utility array.
Purpose of u1 is to find about the differences of the utility_matrix at the start and at the end.

For every state that is not terminal, I created a state_value_matrix, which is all zeros except for the current state's index.
Then I get the given policy's action for the current state.
Then I calculated the current state's utility for the policy's action.
I need to compare this to the maximum utility I can get from the current state.

If the maximum utility is bigger than the policy's utility, I update the utility_matrix, thus the policy by the maximum utility-action.
And at the end I create another u2 - utility array to check if a change was made. If there was a change, 
meaning that the utility array - policy has not converged yet, I run the same loop again until we reach to a converged policy-utility array.
At the end converged utility_matrix is returned.
"""
    
def policy_eval(policy, reward_matrix, transitional_model, discount_factor):
    utility_matrix = np.zeros(16)
    for i in range(16):
        utility_matrix[i] = reward_matrix[i]
    count = 0
    while True:
        u1=np.zeros(16)
        for i in range(16):
            u1[i] = utility_matrix[i]
        count += 1
        for state in range(16):
            if state not in[3,5,13,14,15]:
                state_value_matrix = np.zeros((1, 16))
                state_value_matrix[0,state] = 1.0
                state_action = policy[state]
                reward = reward_matrix[state]
                state_utility = find_state_utility_for_action(state_action, utility_matrix, state_value_matrix, transitional_model, reward, discount_factor)
                max_utility = find_state_utility(state_value_matrix, transitional_model, utility_matrix, reward, discount_factor)
                if state_utility <= max_utility:
                    if state_utility == max_utility:
                        utility_matrix[state] = max_utility
                    else:
                        policy[state] = find_the_best_action_for_a_state_with_reward(utility_matrix, state_value_matrix, transitional_model, reward, discount_factor)
                        utility_matrix[state] = max_utility
        u2=np.zeros(16)
        for i in range(16):
            u2[i] = utility_matrix[i]
        if np.array_equal(u1, u2):
            break
    return np.array(utility_matrix)

"""
This function finds and returns the utility of the state for the given action.
"""
def find_state_utility_for_action(state_action, utility_matrix, state_value_matrix, transitional_model, reward, discount_factor):
    state_utility_value = np.sum(np.multiply(utility_matrix, np.dot(state_value_matrix, transitional_model[:,:,state_action])))
    return reward + discount_factor * state_utility_value

"""
This function finds and returns the maximum utility of the state for any action.
"""
def find_state_utility(state_value_matrix, transitional_model, utility_matrix, reward, discount_factor):
    action_array = np.array([0.0,0.0,0.0,0.0])#north-east-south-west
    for i in range(4):
        action_array[i] = np.sum(np.multiply(utility_matrix, np.dot(state_value_matrix, transitional_model[:,:,i])))
    return reward + discount_factor * np.max(action_array)

"""
This function finds and returns the action index for which the maximum utility is achieved.
"""
def find_the_best_action_for_a_state_with_reward(utility_array, state_array, transitional_model, reward, discount_factor):
    action_array = np.array([0.0,0.0,0.0,0.0])#north-east-south-west
    for i in range(4):
        action_array[i] = reward + discount_factor * np.sum(np.multiply(utility_array, np.dot(state_array, transitional_model[:,:,i])))
    return action_array.argmax()  
    
"""
Basic printing function that prints a policy in a readable form
"""
def print_policy(policy):
    print(policy[0] + "  " + policy[1] + "  " + policy[2] + "  " + policy[3])
    print(policy[4] + "  " + policy[5] + "  " + policy[6] + "  " + policy[7])
    print(policy[8] + "  " + policy[9] + "  " + policy[10] + "  " + policy[11])
    print(policy[12] + "  " + policy[13] + " " + policy[14:17] + " " + policy[17:])

p = 0.5
p1 = (1-p)/2
iteration_count = 1000
state_reward = -0.01
discount_factor = 0.9

transitional_matrix = create_transitional_matrix(16,4)    

state_reward_matrix_for_value_iteration = create_reward_matrix(state_reward)
value_iteration_utility_matrix = find_utility_matrix_for_value_iteration(iteration_count, state_reward_matrix_for_value_iteration, transitional_matrix, discount_factor)
print("Utility matrix of Value Iteration Algorithm for: \nP = " + str(p) + "\nIteration count = " + str(iteration_count) + "\nState reward = " + str(state_reward) + "\nDiscount factor = " + str(discount_factor) + "\n")
print_utility_matrix(value_iteration_utility_matrix) 
policy = create_policy_string_from_utility_array(value_iteration_utility_matrix) 
#policy = create_policy_for_value_iteration(value_iteration_utility_matrix, transitional_matrix, state_reward_matrix_for_value_iteration)
print("\nPolicy created by the value iteration:\n")
print_policy(policy)


print("\n\n\nUtility matrix of Policy Iteration Algorithm for: \nP = " + str(p) + "\nState reward = " + str(state_reward) + "\nDiscount factor = " + str(discount_factor) + "\n")
dummy_policy = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
state_reward_matrix_for_policy_iteration = create_reward_matrix(state_reward)
policy_iteration_utility_matrix = policy_eval(dummy_policy, state_reward_matrix_for_policy_iteration, transitional_matrix, discount_factor)
print_utility_matrix(policy_iteration_utility_matrix)  
policy_string_for_policy_iteration = create_policy_string_from_utility_array(policy_iteration_utility_matrix)
print("\nPolicy created by the policy iteration:\n")
print_policy(policy_string_for_policy_iteration)

#Start of Q-learning
"""
This big function determines what the next state would be if we know the current state and the action.
Except for s5 and s9, all the states have a unique condition. So I had to do a big function like this
p = probability
p1 = (1-p)/2
"""
def simulate_next_state(current_state, action_index):#0-n, 1-e, 2-s, 3-w
    random_number = np.random.random()
    next_state = -1
    if current_state == 0:
        if action_index == 0:
            if random_number < p + p1:
                next_state = 0
            else:
                next_state = 1
        elif action_index == 1:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 4
            else:
                next_state = 1
        elif action_index == 2:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 1
            else:
                next_state = 4
        else:
            if random_number < p + p1:
                next_state = 0
            else:
                next_state = 4
    elif current_state == 1:
        if action_index == 0:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 2
            else:
                next_state = 1
        elif action_index == 1:
            if random_number < p1 + p1:
                next_state = 1
            else:
                next_state = 2
        elif action_index == 2:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 2
            else:
                next_state = 1
        else:
            if random_number < p1 + p1:
                next_state = 1
            else:
                next_state = 0
    elif current_state == 2:
        if action_index == 0:
            if random_number < p1:
                next_state = 1
            elif random_number < p1 + p1:
                next_state = 3
            else:
                next_state = 2
        elif action_index == 1:
            if random_number < p1:
                next_state = 2
            elif random_number < p1 + p1:
                next_state = 6
            else:
                next_state = 3
        elif action_index == 2:
            if random_number < p1:
                next_state = 1
            elif random_number < p1 + p1:
                next_state = 3
            else:
                next_state = 6
        else:
            if random_number < p1:
                next_state = 2
            elif random_number < p1 + p1:
                next_state = 6
            else:
                next_state = 1
    elif current_state == 4:#s3
        if action_index == 0:
            if random_number < p1 + p1:
                next_state = 4
            else:
                next_state = 0
        elif action_index == 1:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 8
            else:
                next_state = 4
        elif action_index == 2:
            if random_number < p1 + p1:
                next_state = 4
            else:
                next_state = 8
        else:
            if random_number < p1:
                next_state = 0
            elif random_number < p1 + p1:
                next_state = 8
            else:
                next_state = 4
    elif current_state == 6:#s4
        if action_index == 0:
            if random_number < p1:
                next_state = 6
            elif random_number < p1+ p1:
                next_state = 7
            else:
                next_state = 2
        elif action_index == 1:
            if random_number < p1:
                next_state = 2
            elif random_number < p1 + p1:
                next_state = 10
            else:
                next_state = 7
        elif action_index == 2:
            if random_number < p1:
                next_state = 6
            elif random_number < p1+ p1:
                next_state = 7
            else:
                next_state = 10
        else:
            if random_number < p1:
                next_state = 2
            elif random_number < p1 + p1:
                next_state = 10
            else:
                next_state = 6
    elif current_state == 7:#s4
        if action_index == 0:
            if random_number < p1:
                next_state = 6
            elif random_number < p1+ p1:
                next_state = 7
            else:
                next_state = 3
        elif action_index == 1:
            if random_number < p1:
                next_state = 3
            elif random_number < p1 + p1:
                next_state = 11
            else:
                next_state = 7
        elif action_index == 2:
            if random_number < p1:
                next_state = 6
            elif random_number < p1+ p1:
                next_state = 7
            else:
                next_state = 11
        else:
            if random_number < p1:
                next_state = 3
            elif random_number < p1 + p1:
                next_state = 11
            else:
                next_state = 6
    elif current_state == 8:#s6
        if action_index == 0:
            if random_number < p1:
                next_state = 8
            elif random_number < p1+ p1:
                next_state = 9
            else:
                next_state = 4
        elif action_index == 1:
            if random_number < p1:
                next_state = 4
            elif random_number < p1 + p1:
                next_state = 12
            else:
                next_state = 9
        elif action_index == 2:
            if random_number < p1:
                next_state = 8
            elif random_number < p1+ p1:
                next_state = 9
            else:
                next_state = 12
        else:
            if random_number < p1:
                next_state = 4
            elif random_number < p1 + p1:
                next_state = 12
            else:
                next_state = 8
    elif current_state == 9:#s7
        if action_index == 0:
            if random_number < p1:
                next_state = 8
            elif random_number < p1+ p1:
                next_state = 10
            else:
                next_state = 9
        elif action_index == 1:
            if random_number < p1:
                next_state = 9
            elif random_number < p1 + p1:
                next_state = 13
            else:
                next_state = 10
        elif action_index == 2:
            if random_number < p1:
                next_state = 8
            elif random_number < p1+ p1:
                next_state = 10
            else:
                next_state = 13
        else:
            if random_number < p1:
                next_state = 9
            elif random_number < p1 + p1:
                next_state = 13
            else:
                next_state = 8    
    elif current_state == 10:#s8
        if action_index == 0:
            if random_number < p1:
                next_state = 9
            elif random_number < p1+ p1:
                next_state = 11
            else:
                next_state = 6
        elif action_index == 1:
            if random_number < p1:
                next_state = 6
            elif random_number < p1 + p1:
                next_state = 14
            else:
                next_state = 11
        elif action_index == 2:
            if random_number < p1:
                next_state = 9
            elif random_number < p1+ p1:
                next_state = 11
            else:
                next_state = 14
        else:
            if random_number < p1:
                next_state = 6
            elif random_number < p1 + p1:
                next_state = 14
            else:
                next_state = 9
    elif current_state == 11:#s9
        if action_index == 0:
            if random_number < p1:
                next_state = 11
            elif random_number < p1+ p1:
                next_state = 10
            else:
                next_state = 7
        elif action_index == 1:
            if random_number < p1:
                next_state = 7
            elif random_number < p1 + p1:
                next_state = 15
            else:
                next_state = 11
        elif action_index == 2:
            if random_number < p1:
                next_state = 11
            elif random_number < p1+ p1:
                next_state = 10
            else:
                next_state = 15
        else:
            if random_number < p1:
                next_state = 7
            elif random_number < p1 + p1:
                next_state = 15
            else:
                next_state = 10
    elif current_state == 12:#s10
        if action_index == 0:
            if random_number < p1:
                next_state = 12
            elif random_number < p1+ p1:
                next_state = 13
            else:
                next_state = 8
        elif action_index == 1:
            if random_number < p1:
                next_state = 8
            elif random_number < p1 + p1:
                next_state = 12
            else:
                next_state = 13
        elif action_index == 2:
            if random_number < p + p1:
                next_state = 12
            else:
                next_state = 13
        else:
            if random_number < p1 + p:
                next_state = 12
            else:
                next_state = 8
    return next_state

"""
Depending on the random float we get from numpy, either chooses a random action to explore the environment 
or gets the maximum valued action for the current state
"""
def choose_action(current_state, q_value_table):
    if np.random.random() < exploration_probability:
        return np.random.randint(0,4)
    else:
        return q_value_table[current_state].argmax()


"""
Returns the terminal states' reward, if not a terminal state returns 0
"""
def find_state_reward(state):
    if state == 3:
        return 1
    elif state == 13:
        return 1
    elif state == 14:
        return -10
    elif state == 15:
        return 10
    else:
        return -0.01
"""
Main Q learning function. 
For iteration_count times, 
    Selects a new action for the current state. This selection depends on the exploration probability. If agent chooses to explore a random action is returned.
    If agent decides to exploit, using the q table, maximum rewarded action is returned.
    Depending on the new action, new state is simulated.
    Q table is updated using the formula with learning rate, reward and discount factor.
    if the reward agent gets from the moved state is 0, meaning that agent is still not at a terminal state since only terminal states have a reward of 0, current state gets updated to the moved state.
    if the reward is not 0, agent moved to a terminal state, which means agent has to start over, thus the current state is now 8.
"""
def update_q_value(q_value_table, current_state, iteration_count):
    for i in range(iteration_count):
        new_action = choose_action(current_state, q_value_table)
        new_state = simulate_next_state(current_state, new_action)
        learning_rate_dictionary[str(current_state)+","+str(new_action)] = learning_rate_dictionary[str(current_state)+","+str(new_action)] + 1
        count = learning_rate_dictionary[str(current_state)+","+str(new_action)]
        learning_rate = 1/count
        reward = find_state_reward(new_state)
        q_value_table[current_state, new_action] = (1-learning_rate)*q_value_table[current_state, new_action] + learning_rate * (reward + discount_factor * np.max(q_value_table[new_state, :]))   
        if reward == -0.01:
            current_state = new_state
        else:
            current_state = 8

"""
Gets the best action values from the table, utility matrix of q table.
"""
def get_best_values_from_q_table(q_table):
    best_values = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for state in range(16):
        best_values[state] = q_table[state].max()
    return best_values

"""
Returns the policy string from the q table
"""
def get_policy_from_q_table(q_table):
    policy = ""
    for i in range(16):
        if i == 3 or i == 13:
            policy += "1"
        elif i == 5:
            policy += "X"
        elif i == 14:
            policy += "-10"
        elif i == 15:
            policy += "10"
        else:            
            best_move_index = q_table[i].argmax()
            if best_move_index == 0:
                policy += "^"
            elif best_move_index == 1:
                policy += ">"
            elif best_move_index == 2:
                policy += "v"
            else:
                policy += "<"
    return policy


"""
Basic printing function that prints the best actions in a readable manner
"""
def print_best_values_from_q_table(q_table):
    best = get_best_values_from_q_table(q_table)
    print(best[0:4])
    print(best[4:8])
    print(best[8:12])
    print(best[12:16])

def create_dictionary_for_one_over_count_learning_rate():
    res = "{"
    for i in range(16):
        for j in range(4):
            res += "\""+str(i) +","+str(j)+ "\":0," 
    res += "}"
    return res

p = 0.8
p1 = (1-p)/2
discount_factor = 0.8
learning_rate_dictionary = {"0,0":0,"0,1":0,"0,2":0,"0,3":0,"1,0":0,"1,1":0,"1,2":0,"1,3":0,"2,0":0,"2,1":0,"2,2":0,"2,3":0,"3,0":0,"3,1":0,"3,2":0,"3,3":0,"4,0":0,"4,1":0,"4,2":0,"4,3":0,"5,0":0,"5,1":0,"5,2":0,"5,3":0,"6,0":0,"6,1":0,"6,2":0,"6,3":0,"7,0":0,"7,1":0,"7,2":0,"7,3":0,"8,0":0,"8,1":0,"8,2":0,"8,3":0,"9,0":0,"9,1":0,"9,2":0,"9,3":0,"10,0":0,"10,1":0,"10,2":0,"10,3":0,"11,0":0,"11,1":0,"11,2":0,"11,3":0,"12,0":0,"12,1":0,"12,2":0,"12,3":0,"13,0":0,"13,1":0,"13,2":0,"13,3":0,"14,0":0,"14,1":0,"14,2":0,"14,3":0,"15,0":0,"15,1":0,"15,2":0,"15,3":0,}
#learning_rate = 
exploration_probability = 0.2
iteration_count = 39850

q_table = np.zeros((16,4)) #0-n, 1-e, 2-s, 3-w
update_q_value(q_table, 8, iteration_count)

print("\n\n\nQ table for: \nP = " + str(p) + "\nDiscount factor = " + str(discount_factor) + "\nLearning rate = " + str("1/count") + "\nExploration probability = " + str(exploration_probability) + "\nIteration count = " + str(iteration_count) + "\n")
print(q_table)
print("\n\nBest values for each state:")
print_best_values_from_q_table(q_table)
print("\n\nPolicy of the Q table" )
policy = get_policy_from_q_table(q_table)
print_policy(policy)