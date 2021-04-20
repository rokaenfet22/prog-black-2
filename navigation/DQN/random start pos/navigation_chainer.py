import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import math

"""
state = [cx,cy,tx,ty]

+++XG
+X+++
+XXX+
+++X+
S++X+

+ = open space
X = wall
S = start
G = Goal
"""

#walls = [[0,3],[1,1],[2,1],[3,3],[2,3],[2,2]]
walls = [[2,2],[2,1],[2,3],[3,3],[4,3],[1,1],[0,3]]
walls= [[0,1],[0,3],[2,1],[2,3],[4,1],[4,3]]
walls=[[2,2],[2,1],[2,3]]
walls=[]

class Nav(chainer.Chain):
    def __init__(self,obs_size,n_actions,n_hidden_channels):
        super(Nav,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l4=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False):
        h1=F.sigmoid(self.l1(x))
        h2=F.sigmoid(self.l2(h1))
        h3=F.sigmoid(self.l3(h2))
        y=chainerrl.action_value.DiscreteActionValue(self.l4(h3)) #agent chooses distinct action from finite action set A
        return y

def get_dist(s):
    return math.sqrt((s[0]-s[2])**2+(s[1]-s[3])**2)

def get_direc(s): #get state representation of direction from ay,ax to by,bx
    [ay,ax]=s[:2]
    [by,bx]=s[2:4]
    #coords r in pygame style
    s=0
    if ax==bx:
        if ay>by:s=1 #N
        else:s=5 #S
    elif ay==by:
        if ax<bx:s=3 #E
        else:s=7 #W
    elif ax>bx:
        if ay>by:s=8 #NW
        else:s=6 #SW
    elif ax<bx:
        if ay>by:s=2 #NE
        else:s=4 #SE
    return s

def find_index(arr,t): #finding index within array of np.array
    for i,n in enumerate(arr):
        if np.array_equal(t,n):
            return i
    return False

def disp_wall():
    grid=[["+" for a in range(5)] for _ in range(5)]
    for y,x in walls:
        grid[y][x]="X"
    grid[4][0]="S"
    grid[0][4]="G"
    for n in grid:
        print("".join(n))

def random_action():
    return np.random.choice([0,1,2,3])

def setup(gamma,obs_size,n_actions,n_hidden_channels,start_epsilon=1,end_epsilon=0.1,num_episodes=1):
    func=Nav(obs_size,n_actions,n_hidden_channels) #model
    optimizer=chainer.optimizers.Adam(eps=1e-8)
    optimizer.setup(func)
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=start_epsilon,end_epsilon=end_epsilon,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10**6)
    phi=lambda x: x.astype(np.float32,copy=False)
    agent=chainerrl.agents.DQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    #agent=chainerrl.agents.DoubleDQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    return agent

def step(s,a):
    r=0
    moved=False
    #record pos
    py,px=s[0],s[1]
    if a==0 and s[0]!=0:#up
        s[0]-=1
        moved=True
    elif a==1 and s[0]!=4:#down
        s[0]+=1
        moved=True
    elif a==2 and s[1]!=0:#left
        s[1]-=1
        moved=True
    elif a==3 and s[1]!=4:#right
        s[1]+=1
        moved=True
    else: #if no movement observed, penalize
        r=-100
    if moved and [s[0],s[1]] in walls: s[0],s[1]=py,px; r=-100 #if collided into wall, penalize
    return s,r

def train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100):
    #agent setup below
    agent=setup(0.8,4,4,50,1,0.1,num_episodes) #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01,num_episodes=1
    #loading prexisting models
    if load:
        agent.load("nav_agent"+name)

    #step limit
    max_step=100

    for episode in range(num_episodes):
        state=np.array([4,0,0,4]) #pos
        
        #for random opponent spawn
        state=r_runner_pos(state)

        start_state=[n for n in state]
        r=0
        total_r=0
        step_taken=0
        #distance
        dist=get_dist(state)
        og_dist=get_dist(state)
        longest_dist=get_dist(np.array([0,4,4,0]))
        caught=True
        while [state[0],state[1]]!=[state[2],state[3]] and caught:
            action=agent.act_and_train(state,r)
            state,r=step(state,action)
            #distance rewarding
            d=get_dist(state)
            if d<dist:
                r+=longest_dist-d
            elif d>dist:
                r-=longest_dist
            dist=d

            total_r+=r
            step_taken+=1

            #time out
            if step_taken>max_step:
                r-=100
                print(state)
                caught=False
        if caught:
            r+=10**6
        total_r+=r
        agent.stop_episode_and_train(state,r,caught)
        #if episode%disp_interval==0:
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}")
            print()
        if episode%save_interval==0 and interv_save:
            agent.save("nav_agent"+name)
    if save:
        agent.save("nav_agent"+name)
    print(agent.get_statistics())

def visualize(num_recs,name):
    #setup model like in train
    agent=setup(0.8,4,4,50,0,0,1) #start,end epsilon = 0,0 for no random_action

    agent.load("nav_agent"+name)

    record=[]
    for _ in range(num_recs):
        state=np.array([4,0,0,4]) #pos


        #choose state
        state=r_both_pos()

        t_rec=[]
        while [state[0],state[1]]!=[state[2],state[3]]:
            t_rec.append([n for n in state])
            #action=agent.act(state)
            action=agent.act_and_train(state,0)
            state,r=step(state,action)
            #print(f"state={state}, actions={action}")
        t_rec.append([n for n in state])
        agent.stop_episode()
        #agent.load("nav_agent"+name)
        record.append(t_rec)
        print(len(t_rec))
    
    import pygame
    window_x,window_y=300,300

    px=window_x//5

    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    for recs in record:
        
        #print(len(recs))
        while len(recs)>0:
            pygame.time.delay(20)
            
            window.fill(pygame.Color("black"))

            #drawing everything goes here
            for n in walls:
                draw_rect([n[1]*px,n[0]*px,px,px],"white")
            a=recs.pop(0)
            draw_rect([a[1]*px,a[0]*px,px,px],"red",0) #chaser
            draw_rect([a[3]*px,a[2]*px,px,px],"green",0) #runner

            pygame.display.flip()
        pygame.time.delay(100)
    pygame.quit()

def sep_visualize(num_recs,name):
    #setup model like in train
    agent=setup(0.3,5,4,50,0,0,1) #start,end epsilon = 0,0 for no random_action

    agent.load("nav_agent"+name)

    record=[]
    for _ in range(num_recs):
        state=np.array([4,0,0,4,0]) #pos

        #set starting pos
        state=r_both_pos(state)
        #state=r_chaser_pos(state)

        #set initial dist and direc
        state[4]=get_direc(state[:4])
        #state[5]=get_dist(state[:4])

        t_rec=[]
        while [state[0],state[1]]!=[state[2],state[3]]:
            t_rec.append([n for n in state])
            action=agent.act(state)
            #action=agent.act_and_train(state,0)
            state,r=step(state,action)
            state[4]=get_direc(state[:4])
            #state[5]=get_dist(state[:4])
            #print(f"state={state}, actions={action}")
        t_rec.append([n for n in state])
        agent.stop_episode()
        #agent.load("nav_agent"+name)
        record.append(t_rec)
        print(len(t_rec))
    
    import pygame
    window_x,window_y=300,300

    px=window_x//5

    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    for recs in record:
        
        #print(len(recs))
        while len(recs)>0:
            pygame.time.delay(20)
            
            window.fill(pygame.Color("black"))

            #drawing everything goes here
            for n in walls:
                draw_rect([n[1]*px,n[0]*px,px,px],"white")
            a=recs.pop(0)
            draw_rect([a[1]*px,a[0]*px,px,px],"red",0) #chaser
            draw_rect([a[3]*px,a[2]*px,px,px],"green",0) #runner

            pygame.display.flip()
        pygame.time.delay(0)
    pygame.quit()

def r_chaser_pos(state):
    state[0]=np.random.choice([0,1,2,3,4])
    state[1]=np.random.choice([0,1,2,3,4])
    while [state[0],state[1]]==[state[2],state[3]]:
        state[0]=np.random.choice([0,1,2,3,4])
        state[1]=np.random.choice([0,1,2,3,4])
    return state

def r_runner_pos(state):
    state[2]=np.random.choice([0,1,2,3,4])
    state[3]=np.random.choice([0,1,2,3,4])
    while [state[0],state[1]]==[state[2],state[3]]:
        state[2]=np.random.choice([0,1,2,3,4])
        state[3]=np.random.choice([0,1,2,3,4])
    return state

def r_both_pos(state): #random pos for both chaser and runner
    state[0]=np.random.choice([0,1,2,3,4])
    state[1]=np.random.choice([0,1,2,3,4])
    state[2]=np.random.choice([0,1,2,3,4])
    state[3]=np.random.choice([0,1,2,3,4])
    while [state[0],state[1]]==[state[2],state[3]] or [n for n in state[:2]] in walls or [n for n in state[2:4]] in walls:
        state[0]=np.random.choice([0,1,2,3,4])
        state[1]=np.random.choice([0,1,2,3,4])
        state[2]=np.random.choice([0,1,2,3,4])
        state[3]=np.random.choice([0,1,2,3,4])
    return state

def one_train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100):
    #agent setup below
    agent=setup(0.8,6,4,50,1,0.1,num_episodes) #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01,num_episodes=1
    #loading prexisting models
    if load:
        agent.load("nav_agent"+name)

    #step limit
    max_step=100

    for episode in range(num_episodes):
        state=np.array([4,0,0,4,0,0]) #cx,cy,rx,ry,dist,direc
        
        #set starting pos
        state=r_both_pos(state)

        #set initial dist and direc
        state[4]=get_dist(state[:4])
        state[5]=get_direc(state[:4])

        start_state=[n for n in state]
        r=0
        total_r=0
        step_taken=0
        #distance
        dist=get_dist(state)
        og_dist=get_dist(state)
        longest_dist=get_dist(np.array([0,4,4,0]))
        caught=True
        while [state[0],state[1]]!=[state[2],state[3]] and caught:
            action=agent.act_and_train(state,r)
            state,r=step(state,action)
            #distance rewarding
            d=get_dist(state)
            if d<dist:
                r+=longest_dist
            elif d>dist:
                r-=longest_dist
            dist=d

            total_r+=r #update reward
            step_taken+=1 #update step counter

            #update dist and direc
            state[4]=get_dist(state[:4])
            state[5]=get_direc(state[:4])

            #time out
            if step_taken>max_step:
                r-=100
                print(state)
                caught=False
        if caught:
            r+=10**5
        total_r+=r
        agent.stop_episode_and_train(state,r,caught)
        #if episode%disp_interval==0:
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}")
            print()
        if episode%save_interval==0 and interv_save:
            agent.save("nav_agent"+name)
    if save:
        agent.save("nav_agent"+name)
    print(agent.get_statistics())

def two_train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100):
    #agent setup below
    agent=setup(0.3,5,4,50,1,0.1,num_episodes) #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01,num_episodes=1
    #loading prexisting models
    if load:
        agent.load("nav_agent"+name)

    #longest possible dist
    longest_dist=get_dist(np.array([0,4,4,0]))
    #step limit
    max_step=30
    #rmbr time out states
    timeout_states=[]
    #prev pos buffer quantity
    mem=10

    for episode in range(num_episodes):
        state=np.array([4,0,0,4,0]) #cx,cy,rx,ry,direc,dist
        
        #choose from either timeout_states, or other option
        if len(timeout_states)>0:
            if np.random.uniform(0,1)>0.5:
                state=r_both_pos(state)
                #state=r_chaser_pos(state)
            else:
                print("-----------------------------------------------")
                state=np.copy(timeout_states.pop(0))
        else:
            state=r_both_pos(state)

        #set initial dist and direc
        state[4]=get_direc(state[:4])
        #state[5]=get_dist(state[:4])

        #rmbr previous state
        prev_states=[]

        #for debugging
        start_state=[n for n in state]

        #initial vals
        r=0
        total_r=0
        step_taken=0

        #distance
        dist=get_dist(state)
        og_dist=get_dist(state)

        #caught or not
        caught=True

        while [state[0],state[1]]!=[state[2],state[3]] and caught: #step loop
            action=agent.act_and_train(state,r) #get action from agent, and update network

            #recording past states, [newest,...,oldest]
            if len(prev_states)<=mem: 
                prev_states.insert(0,state[:4])
            else:
                prev_states.pop()
                prev_states.insert(0,state[:4])

            state,r=step(state,action) #update state from given action

            #distance rewarding and update
            d=get_dist(state)
            if d<dist: r+=longest_dist
            elif d>dist: r-=longest_dist
            dist=d

            #penalizing for repeated pos in previous "mem" steps
            i=find_index(prev_states,state[:4])
            if i: r-=longest_dist*((mem-i+1)*(5/mem))

            total_r+=r #update reward
            step_taken+=1 #update step counter

            #update dist and direc
            state[4]=get_direc(state[:4])
            #state[5]=get_dist(state[:4])

            #time limit
            if step_taken>max_step:
                r-=100
                print("***********************************************")
                timeout_states.append(np.copy(start_state))
                caught=False
        if caught:
            r+=10**5
        total_r+=r
        agent.stop_episode_and_train(state,r,caught)
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}, start state: {start_state}, length of timeouts: {len(timeout_states)}")
            print()
        if episode%save_interval==0 and interv_save:
            agent.save("nav_agent"+name)
    if save:
        agent.save("nav_agent"+name)
    print(agent.get_statistics())

def wall_train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100):
    #agent setup below
    agent=setup(0.3,5,4,50,1,0.1,num_episodes) #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01,num_episodes=1
    #loading prexisting models
    if load:
        agent.load("nav_agent"+name)

    longest_dist=get_dist(np.array([0,4,4,0])) #longest possible dist
    max_step=round(longest_dist*10) #step limit
    timeout_states=[] #rmbr time out states
    mem=10 #prev pos buffer quantity
    start_state_choice_p=0.5 #probability for choosing starting state as random, else from timeout_states

    for episode in range(num_episodes):
        state=np.array([4,0,0,4,0]) #cx,cy,rx,ry,direc,dist
        
        #choose from either timeout_states, or other option
        if len(timeout_states)>0:
            if np.random.uniform(0,1)>start_state_choice_p:
                #state=r_both_pos(state)
                #state=r_chaser_pos(state)
                #state=r_runner_pos(state)
                pass
            else:
                #print("-----------------------------------------------")
                state=np.copy(timeout_states.pop(0))
        else:
            #state=r_both_pos(state)
            #state=r_chaser_pos(state)
            #state=r_runner_pos(state)
            pass

        #set initial dist and direc
        state[4]=get_direc(state[:4])
        #state[5]=get_dist(state[:4])

        #rmbr previous state
        prev_states=[]

        #for debugging
        start_state=[n for n in state]

        #initial vals
        r=0
        total_r=0
        step_taken=0

        #distance
        dist=get_dist(state)
        og_dist=get_dist(state)

        #caught or not
        caught=True

        while [state[0],state[1]]!=[state[2],state[3]] and caught: #step loop
            action=agent.act_and_train(state,r) #get action from agent, and update network

            #recording past states, [newest,...,oldest]
            if len(prev_states)<=mem:
                prev_states.insert(0,state[:4])
            else:
                prev_states.pop()
                prev_states.insert(0,state[:4])

            state,r=step(state,action) #update state from given action

            #euclidean distance rewarding and update
            d=get_dist(state)
            if d<dist: r+=longest_dist
            elif d>dist: r-=longest_dist
            dist=d

            #penalizing for repeated pos in previous "mem" steps
            i=find_index(prev_states,state[:4])
            if i: r-=longest_dist*(mem-i)*(5/mem)

            total_r+=r #update reward
            step_taken+=1 #update step counter

            #update dist and direc
            state[4]=get_direc(state[:4])
            #state[5]=get_dist(state[:4])

            #time limit
            if step_taken>max_step:
                r-=100
                #print("***********************************************")
                timeout_states.append(np.copy(start_state))
                caught=False
        if caught:
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            r+=10**7
        total_r+=r
        agent.stop_episode_and_train(state,r,caught)
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}, start state: {start_state}, length of timeouts: {len(timeout_states)}")
            print()
        if episode%save_interval==0 and interv_save:
            agent.save("nav_agent"+name)
    if save:
        agent.save("nav_agent"+name)
    print(agent.get_statistics())

def wall_visualize(num_recs,name):
    #setup model like in train
    agent=setup(0.3,5,4,50,0,0,1) #start,end epsilon = 0,0 for no random_action
    agent.load("nav_agent"+name)
    record=[]
    for _ in range(num_recs):
        state=np.array([4,0,0,4,0]) #pos

        #set starting pos
        #state=r_both_pos(state)
        #state=r_chaser_pos(state)
        #state=r_runner_pos(state)

        #set initial dist and direc
        state[4]=get_direc(state[:4])
        #state[5]=get_dist(state[:4])

        t_rec=[]
        while [state[0],state[1]]!=[state[2],state[3]]:
            t_rec.append([n for n in state])
            action=agent.act(state)
            #action=agent.act_and_train(state,0)
            state,r=step(state,action)
            state[4]=get_direc(state[:4])
            #state[5]=get_dist(state[:4])
            #print(f"state={state}, actions={action}")
        t_rec.append([n for n in state])
        agent.stop_episode()
        #agent.load("nav_agent"+name)
        record.append(t_rec)
        print(len(t_rec))
    
    import pygame
    window_x,window_y=300,300

    px=window_x//5

    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    for recs in record:
        #print(len(recs))
        while len(recs)>0:
            pygame.time.delay(20)
            window.fill(pygame.Color("black"))
            #drawing everything goes here
            for n in walls:
                draw_rect([n[1]*px,n[0]*px,px,px],"white")
            a=recs.pop(0)
            draw_rect([a[1]*px,a[0]*px,px,px],"red",0) #chaser
            draw_rect([a[3]*px,a[2]*px,px,px],"green",0) #runner
            pygame.display.flip()
        pygame.time.delay(0)
    pygame.quit()

disp_wall()
input()
wall_train(5000,"10",True,False) #(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100)
wall_visualize(50,"10")

"""using one train"""
#1 = [4,0,0,4] 3k eps
#2 = [4,0,r,r] 5k eps
#3 = [r,r,0,4] 4k eps accidentally saved over it ;p
#4 layers DQN, gamma=0.8, actvifunc=tanh, adam, prioritizedbufferreplay, start_replay=300, target_update_interval=50x
"""using two train"""
#4 = [r,r,r,r]
#5 layers DQN, gamma=0.8, activfunc=relu, adam, prioritizedbufferreplay, satrt_replay=300, target_update_interval=50
#5 = [r,r,r,r] 100k eps
#tanh, added distance and direction into states
#6 penalize for positions repeated frequently (memory buffer size="mem")
#7 WORKING sigmoid, no walls, [0.3,5,4,50,1,0.1] setup params, adam, lineardecayepsilongreedy, prioritizebufferreplay, DQN
"""using wall train"""
#8 = [4,0,0,4] <500 eps walls = [[2,2],[2,1],[2,3]]
#9 = [4,0,0,4] >5k eps walls = [[2,2],[2,1],[2,3],[3,3],[4,3],[1,1],[0,3]]
#10 = walls = [[0,1],[0,3],[2,1],[2,3],[4,1],[4,3]]