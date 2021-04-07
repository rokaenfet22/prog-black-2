"""
Using Q Learning to Solve the "Mouse Learning problem, Skinner box"
There is a mouse and a vending machine with 2 buttons, a power button and product button. The mouse must press the product button when
the vending machine is powered on to recieve treats.
Terminologies
St = State
a = action
r = reward
Q value = Q(St,a), likelihood of choosing action a at state st
max Q = max Q value in resulting state after action

Q(st,a) <- (1-x)Q(st,a) + x(r+y max Q)
Where x and y are constants

2 States, power on/off = 1/0
2 Actions, pressing product btn/power btn = 0/1
Reward when st=1 and a=1 of 1, otherwise 0
"""

import numpy as np

def random_action():
    return np.random.choice([0,1])

def get_action(next_state,episode):
    epsilon=0.5*(1/(episode+1)) #epsilon greedy method. Slowly choosing optimum
    if epsilon<=np.random.uniform(0,1):
        a=np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action=np.random.choice(a)
    else:
        next_action=random_action()
    return next_action

def step(state,action):
    reward=0
    if state==0:
        if action==0:
            state=1
        else:
            state=0
    else:
        if action==0:
            state=0
        else:
            state=1
            reward=1
    return state,reward

def update_Qtable(q_table,state,action,reward,next_state):
    x=0.9
    y=0.5
    next_maxQ=max(q_table[next_state])
    q_table[state,action]=(1-x)*q_table[state,action]+x*(reward+y*next_maxQ)
    return q_table

max_number_of_steps=5 #step num per try
num_episodes=10 #number of tries
q_table=np.zeros((2,2))

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