"""Environment Reference - 5x5 area, fixed start S, Red goal R, Green goal G and Black zones B.
1 Player starts on S, with the aim to reach G. The player must reach G with max reward, penalties are given by passing B and R.

Actions: up,down,left,right = 0,1,2,3
States: See go_to_goal.png
Reward: Out of bounds, B, R, G = 0,0,-1,1
"""

import numpy as np

sa_table=[
[[1],[0],[0],[5]], [[2],[0],[1],[1]], [[3],[1],[2],[2]], [[4],[2],[3],[3]], [[4],[3],[4],[9]],
[[5],[5],[0],[10]], [[7],[5],[1],[11]], [[8],[6],[2],[12]], [[9],[7],[3],[13]], [[9],[9],[4],[14]],
[[10],[10],[5],[15]], [[12],[10],[6],[16]], [[12],[12],[12],[12]], [[14],[12],[8],[18]], [[14],[15],[9],[19]],
[[15],[15],[10],[20]], [[17],[15],[11],[21]], [[18],[16],[12],[22]], [[19],[17],[13],[23]], [[19],[19],[14],[24]],
[[21],[20],[15],[20]], [[22],[20],[21],[21]], [[23],[21],[22],[22]], [[24],[22],[23],[23]], [[24],[23],[19],[24]]
]

def random_action():
    return np.random.choice([0,1,2,3])

def get_action(next_state,episode):
    epsilon=0.5*(1/(episode+1)) #epsilon greedy method. Slowly choosing optimum
    if epsilon<=np.random.uniform(0,1):
        a=np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action=np.random.choice(a)
    else:
        next_action=random_action()
    return next_action

def step(s,a):
    r=0
    if sa_table[s][a][0]==15: r=-1
    elif sa_table[s][a][0]==20: r=1
    return sa_table[s][a][0],r

def update_Qtable(q_table,state,action,reward,next_state):
    x=0.9
    y=0.5
    next_maxQ=max(q_table[next_state])
    q_table[state,action]=(1-x)*q_table[state,action]+x*(reward+y*next_maxQ)
    return q_table

num_episodes=20 #number of tries
q_table=np.zeros((25,4))

for episode in range(num_episodes):
    state=0
    episode_reward=0
    step_taken=0
    
    while state!=20: #one try's iteration
        action=get_action(state,episode)
        next_state,reward=step(state,action)
        #print(f"state={state+1}, action={action}")
        episode_reward+=reward #add reward
        q_table=update_Qtable(q_table,state,action,reward,next_state)
        state=next_state
        step_taken+=1 #count the number steps needed to reach goal
    
    print(f"episode: {episode+1}, total reward: {episode_reward}")
    print(f"steps taken: {step_taken}")
    #print(q_table)
    print()