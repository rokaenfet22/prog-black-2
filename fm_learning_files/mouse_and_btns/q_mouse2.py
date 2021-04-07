"""
q_mouse.py, except there are now 3 buttons
Button C pressed when A and B are on for reward

states = (Aoff,Boff),(Aon,Boff),(Aoff,Bon),(Aon,Bon) = 0,1,2,3
actions = Pressing A,B,C = 0,1,2
reward = 1 when st=3 and a=2 otherwise 0
"""

import numpy as np

sa_table=[[1,2,0],[0,3,1],[3,0,2],[2,1,3]] #table for states to actions

def random_action():
    return np.random.choice([0,1,2])

def get_action(next_state,episode):
    epsilon=0.5*(1/(episode+1)) #epsilon greedy method. Slowly choosing optimum
    if epsilon<=np.random.uniform(0,1):
        a=np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action=np.random.choice(a)
    else:
        next_action=random_action()
    return next_action

def step(s,a):
    #states = (Aoff,Boff),(Aon,Boff),(Aoff,Bon),(Aon,Bon) = 0,1,2,3
    #actions = (Aon,Aoff),(Bon,Boff),(Con,Coff) = 0,1,2,3,4,5
    if s==3 and a==2: r=1
    else: r=0
    s=sa_table[s][a]
    return s,r

def update_Qtable(q_table,state,action,reward,next_state):
    x=0.9
    y=0.5
    next_maxQ=max(q_table[next_state])
    q_table[state,action]=(1-x)*q_table[state,action]+x*(reward+y*next_maxQ)
    return q_table

max_number_of_steps=10 #step num per try
num_episodes=100 #number of tries
q_table=np.zeros((4,3))

for episode in range(num_episodes):
    state=0
    episode_reward=0

    for t in range(max_number_of_steps): #one try's iteration
        action=get_action(state,episode)
        next_state,reward=step(state,action)
        print(state,action,reward)
        episode_reward+=reward #add reward
        q_table=update_Qtable(q_table,state,action,reward,next_state)
        state=next_state
    
    print(f"episode: {episode+1} total reward: {episode_reward}")
    print(q_table)
    print()