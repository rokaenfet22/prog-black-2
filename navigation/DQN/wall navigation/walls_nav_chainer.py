import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import math

"""
States: outer wall, wall, runner = 0,1,2    close,far = 0,1 
+ = open space
X = wall
S = start
G = Goal
"""

#walls = [[2,2],[2,1],[2,3],[3,3],[4,3],[1,1],[0,3]]
#walls= [[0,1],[0,3],[2,1],[2,3],[4,1],[4,3]]
walls=[[1,1],[1,3],[3,1],[3,3]]
grid_size=5

setup_param=[0.7,20,4,40,1,0.1] #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01
agent_name="wall_agent"

class Nav(chainer.Chain): #class for the DQN
    def __init__(self,obs_size,n_actions,n_hidden_channels):
        super(Nav,self).__init__()
        with self.init_scope(): #defining the layers
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l4=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False): #activation function = sigmoid
        h1=F.sigmoid(self.l1(x))
        h2=F.sigmoid(self.l2(h1))
        h3=F.sigmoid(self.l3(h2))
        y=chainerrl.action_value.DiscreteActionValue(self.l4(h3)) #agent chooses distinct action from finite action set
        return y

def get_dist(s): #returns euclidiean distance between (s[0],s[1]) and (s[2],s[3])
    return math.sqrt((s[0]-s[2])**2+(s[1]-s[3])**2)

def get_direc(s): #get state representation of direction from (s[0],s[1]) to (s[2],s[3])
    [ay,ax]=s[:2] #[y,x] form
    [by,bx]=s[2:4]
    #coords r in pygame style, top left = [0,0] going down adds positively on the y value i.e. [+1,0]
    #representing cardinal directions in clockwise order, starting with N
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

def find_index(arr,t): #arr.index(t), but with numpy arrays. Returns False if t not in arr
    for i,n in enumerate(arr):
        if np.array_equal(t,n):
            return i
    return False

def disp_wall(s): #ASCII representation of the current grid. See top of code for terminology of letters
    grid=[["+" for a in range(5)] for _ in range(5)]
    for y,x in walls:
        grid[y][x]="X"
    grid[s[0]][s[0]]="S"
    grid[s[1]][s[0]]="G"
    for n in grid:
        print("".join(n))

def random_action(): #returns random action, used by "explorer"
    return np.random.choice([0,1,2,3])

#this is where optimizers, explorers, replay buffers, network structure, and agent type is defined
def setup(gamma,obs_size,n_actions,n_hidden_channels,start_epsilon=1,end_epsilon=0.1,num_episodes=1): #the skeletal structure of my agent.
    func=Nav(obs_size,n_actions,n_hidden_channels) #model's structure defined here
    optimizer=chainer.optimizers.Adam(eps=1e-8) #optimizer chosen
    optimizer.setup(func)
    #explorer setup
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=start_epsilon,end_epsilon=end_epsilon,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10**6) #experience replay buffer setup
    phi=lambda x: x.astype(np.float32,copy=False) #must be float32 for chainer
    #defining network type and setting up agent. Required parameter differs for most networks (e.g. DDPG, AC3, DQN)
    agent=chainerrl.agents.DQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    return agent

#given current state s, and action a from network, returns next state and reward for that action a
def step(s,a):
    r=0
    #record chaser position
    py,px=s[0],s[1]
    if a==0 and s[0]!=0:#up
        s[0]-=1
    elif a==1 and s[0]!=4:#down
        s[0]+=1
    elif a==2 and s[1]!=0:#left
        s[1]-=1
    elif a==3 and s[1]!=4:#right
        s[1]+=1
    else: #if no movement observed, penalize
        r=-10
    #if collided into wall, penalize
    if [s[0],s[1]] in walls: s[0],s[1]=py,px; r=-10
    return s,r #return new state, and reward

cardinal_direcs=[[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]] #in order of N,NE,E,SE,S,SW,W,NW

def update_state(s):
    #get direc and dist states x16
    [cy,cx,ry,rx]=[*s[:4]]
    direc_i,dist_i=4,12
    for dy,dx in cardinal_direcs:
        ty,tx=cy,cx
        while True:
            ty+=dy; tx+=dx
            if [ty,tx] in walls: #wall
                s[direc_i]=1
                break
            elif ty<0 or ty>=grid_size or tx<0 or tx>=grid_size: #out of bounds
                s[direc_i]=0
                break
            elif ty==ry and tx==rx: #runner
                s[direc_i]=2
                break
        s[dist_i]=get_dist([ty-dy,tx-dx,cy,cx])
        direc_i+=1;dist_i+=1
    return s

def r_both_pos(state): #generate random pos for both entities, which is not overlapping with themselves nor the walls
    state[:4]=np.random.randint(0,5,4)
    while [state[0],state[1]]==[state[2],state[3]] or [n for n in state[:2]] in walls or [n for n in state[2:4]] in walls:
        state[:4]=np.random.randint(0,5,4)
    return state

def train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100): #actual train algo
    #agent setup below
    agent=setup(*setup_param,num_episodes) 
    #loading prexisting agent
    if load:
        agent.load(agent_name+name)

    longest_dist=get_dist(np.array([0,4,4,0])) #longest possible dist
    max_step=round(longest_dist*10) #time/step limit
    mem=4 #prev pos record
    timeout_states=[]

    for episode in range(num_episodes): #episode loop
        state=np.array([4,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #cx,cy,rx,ry,cardinal direcs and distance [N.NE.E.SE.S.SW.W.NW , Nd,NEd,Ed,SEd,Sd,SWd,Wd,NWd]

        #choose from either timeout_states, or other option
        if len(timeout_states)>0:
            if np.random.uniform(0,1)>0.5:
                s=r_both_pos(state)
            else:
                #print("-----------------------------------------------")
                state=np.copy(timeout_states.pop(0))
        else:
            s=r_both_pos(state)
        state=update_state(state)

        start_state=[n for n in state] #record starting state

        #set vars
        dist=get_dist(state[:4])
        r=0 #cur reward
        total_r=0 #total reward of that episode
        step_taken=0 #step/time counter
        prev_states=[] #with "mem"
        
        caught=True #caught or not

        #for animation
        t_rec=[]
        while [state[0],state[1]]!=[state[2],state[3]] and caught: #step loop
            t_rec.append(np.copy(state)) #animation
            action=agent.act_and_train(state,r) #get action from agent, and update network (train)
            #recording past states, [newest,...,oldest]
            if len(prev_states)<=mem:
                prev_states.insert(0,state[:4])
            else:
                prev_states.pop()
                prev_states.insert(0,state[:4])
            state,r=step(state,action) #update state'spositions from given action
            state=update_state(state) #update the rest of states

            d=get_dist(state[:4])
            if d<dist:r+=1
            #elif d>dist:r-=10
            dist=d

            i=find_index(prev_states,state[:4])
            if i: r-=(mem-i)*(2/mem)

            total_r+=r #update current episode's total reward
            step_taken+=1 #increment step counter
            #time limit, detect failed catch
            if step_taken>=max_step:
                r-=10 #penalize
                timeout_states.append(np.copy(start_state))
                caught=False
        if caught:
            print("***********************************************")
            r+=10**6
        total_r+=r
        agent.stop_episode_and_train(state,r,caught) #final train of episode
        t_rec.append(np.copy(state)) #animation
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}, start state: {start_state}, length of timeouts: {len(timeout_states)}")
            print()
        #saving agent models in intervals
        if episode%save_interval==0 and interv_save:
            agent.save(agent_name+name)
        #periodical aniamtion
        if animate_interval>0 and episode%animate_interval==0:
            animate([t_rec])
    #saving agent models after all episode ran
    if save:
        agent.save(agent_name+name)
    #display basic values of network
    print(agent.get_statistics())

def animate_record(num_recs,name):
    #setup model like in train
    agent=setup(*setup_param[:4],0,0,1) #start,end epsilon = 0,0 for no random_action
    agent.load(agent_name+name)
    record=[]
    for _ in range(num_recs):
        state=np.array([4,0,0,4,0]) #pos
        s=r_both_pos(s)
        state=update_state(state)
        t_rec=[]
        while [state[0],state[1]]!=[state[2],state[3]]:
            t_rec.append([n for n in state])
            action=agent.act(state)
            #action=agent.act_and_train(state,0)
            state,r=step(state,action)
            state=update_state(state)
            #print(f"state={state}, actions={action}")
        t_rec.append([n for n in state])
        agent.stop_episode()
        record.append(t_rec)
        print(len(t_rec))
    return record

def animate(record):
    import pygame
    window_x,window_y=300,300
    px=window_x//5
    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))
    for recs in record:
        pygame.time.delay(200)
        while len(recs)>0:
            pygame.time.delay(60)
            window.fill(pygame.Color("black"))
            for n in walls:
                draw_rect([n[1]*px,n[0]*px,px,px],"white")
            a=recs.pop(0)
            draw_rect([a[1]*px,a[0]*px,px,px],"red",0) #chaser
            draw_rect([a[3]*px,a[2]*px,px,px],"green",0) #runner
            pygame.display.flip()
    pygame.quit()

train(5001,"1",True,False,True,1,200) #(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100)