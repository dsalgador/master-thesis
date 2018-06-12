import numpy as np

def int_to_action(int_action: int, env):
    """
    So far assumed k = 2, 3 or 4:
    
    Converts an integer between 0 and env.action_space.shape[1]**env.action_space.shape[0]
    which is (n+1)^k where n is the number of tanks and k the number of trucks.
    
    return vect_action: a k-dimensional vector with components in the range 0,...n. 
    For k = 2, vect_action = [i,j] is the action of truck 1 going to tank i and truck 2 going to tank j.
    (i, j = n means staying at the depot, 0,....,n-1 are the real tanks).
    The associated integer is i*(n+1) + j
    
    For k = 3, vect_action = [i,j,l] is the action of truck 1 going to tank i, truck 2 going to tank j,
    and truck 3 going to tank l.
    (i, j, l = n means staying at the depot, 0,....,n-1 are the real tanks).
    The associated integer is (i*(n+1) + j)*(n+1) + l
    
    For k = 4, vect_action = [i,j,l,m] is the action of truck 1 going to tank i, truck 2 going to tank j,
    truck 3 going to tank l and truck 4 going to tank m.
    (i, j, l, m = n means staying at the depot, 0,....,n-1 are the real tanks).
    The associated integer is ((i*(n+1) + j)*(n+1) + l)*(n+1) + m

    """
    nplus1 = env.action_space.shape[1]
    k = env.action_space.shape[0]
    n_actions = nplus1**k
    
    if k == 2:
        j = int_action % nplus1
        i = int((int_action-j)/nplus1)
        vect_action = np.array([i,j])
      
    elif k == 3:
        l = int_action % nplus1
        ij = int( (int_action - l)/nplus1 ) 
        j = ij % nplus1
        i = int((ij-j)/nplus1)
        vect_action = np.array([i,j,l])
        
    elif k == 4:
        m = int_action % nplus1
        ml = int((int_action - m)/nplus1)
        l = ml % nplus1
        ij = int((ml-l)/nplus1) 
        j = ij % nplus1
        i = int((ij-j)/nplus1)
        vect_action = np.array([i,j,l,m])
    
    else:
        raise ValueError("The number of trucks k of the environment is different from 2, 3 or 4")
    return vect_action

def action_to_int(vect_action: np.array, env):
    """
    Assumed k = 2,3 or 4, so vect_action has 2, 3 or components respectively.
    """
    nplus1 = env.action_space.shape[1]
    k = env.action_space.shape[0]
    if k == 2:
        int_action = vect_action[0] * nplus1 + vect_action[1]
    elif k == 3:
        int_action = (vect_action[0] * nplus1 + vect_action[1])*nplus1 + vect_action[2]
    elif k == 4:
        int_action = ((vect_action[0] * nplus1 + vect_action[1])*nplus1 + vect_action[2])*nplus1 + vect_action[3] 
    else:
        raise ValueError("The number of trucks k of the environment is different from 2, 3 or 4")

    return int_action

#####################################################
"""
Adapted from https://github.com/ageron/handson-ml 
"""

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def normalize_rewards(all_discounted_rewards): #, discount_rate):
    #all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
########################################################

def my_indicator(x_min, x, x_max):
            if (x> x_min) & (x<=x_max):
                return 1
            else: 
                return 0
def is_empty(x):
        if x <=0:
            return 1
        else: 
            return 0