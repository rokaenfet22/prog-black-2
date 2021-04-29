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

walls=[[2,2],[2,3],[2,4]]
walls=[]
grid_size=10

chaser_setup_param=[0.3,22,4,50,1,0.1] #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01
runner_setup_param=[0.3,22,4,50,1,0.1]
chaser_agent_name="main_chaser"
runner_agent_name="main_runner"
h_lay_num=20

class Nav(chainer.Chain): #class for the DQN
    def __init__(self,obs_size,n_actions,n_hidden_channels):
        super(Nav,self).__init__()
        with self.init_scope(): #defining the layers
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l4=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False): #activation function = sigmoid
        h1=F.sigmoid(self.l1(x)) #options including tanh, sigmoid, relu, and variants of those too
        h2=F.sigmoid(self.l2(h1))
        h3=F.sigmoid(self.l3(h2))
        y=chainerrl.action_value.DiscreteActionValue(self.l4(h3)) #agent chooses distinct action from finite action set
        return y

def get_dist(s): #returns euclidiean distance between (s[0],s[1]) and (s[2],s[3])
    return math.sqrt((s[0]-s[2])**2+(s[1]-s[3])**2)

def get_direc(s,e): #get state representation of direction
    if e=="chaser":
        [ay,ax]=s[:2] #[y,x] form
        [by,bx]=s[2:4]
    elif e=="runner":
        [ay,ax]=s[2:4]
        [by,bx]=s[:2]
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
    #optimizer=chainer.optimizers.AdaDelta(eps=1e-8)
    optimizer.setup(func)
    #explorer setup
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=start_epsilon,end_epsilon=end_epsilon,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10**6) #experience replay buffer setup
    phi=lambda x: x.astype(np.float32,copy=False) #must be float32 for chainer
    #defining network type and setting up agent. Required parameter differs for most networks (e.g. DDPG, AC3, DQN)
    agent=chainerrl.agents.DQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    #agent=chainerrl.agents.DoubleDQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    #agent=chainerrl.agents.AC3(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    return agent

#given current state s, and action a from network, returns next state and reward for that action a
def step(s,a,e):
    r=0
    if e=="chaser":
        py,px=s[0],s[1]
        if a==0 and s[0]!=0:#up
            s[0]-=1
        elif a==1 and s[0]!=grid_size-1:#down
            s[0]+=1
        elif a==2 and s[1]!=0:#left
            s[1]-=1
        elif a==3 and s[1]!=grid_size-1:#right
            s[1]+=1
        else:
            r-=1
        if [s[0],s[1]] in walls: s[0],s[1]=py,px; r=-1
    elif e=="runner":
        py,px=s[2],s[3]
        if a==0 and s[2]!=0:#up
            s[2]-=1
        elif a==1 and s[2]!=grid_size-1:#down
            s[2]+=1
        elif a==2 and s[3]!=0:#left
            s[3]-=1
        elif a==3 and s[3]!=grid_size-1:#right
            s[3]+=1
        else:
            r-=1
        if [s[2],s[3]] in walls: s[2],s[3]=py,px; r=-1
    return s,r #return new state, and reward

cardinal_direcs=[[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]] #in order of N,NE,E,SE,S,SW,W,NW

def update_state(s,e):
    #get direc and dist states x16
    [cy,cx,ry,rx]=[*s[:4]]
    direc_i,dist_i=4,12
    if e=="chaser":
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
    elif e=="runner":
        for dy,dx in cardinal_direcs:
            ty,tx=ry,rx
            while True:
                ty+=dy; tx+=dx
                if [ty,tx] in walls: #wall
                    s[direc_i]=1
                    break
                elif ty<0 or ty>=grid_size or tx<0 or tx>=grid_size: #out of bounds
                    s[direc_i]=0
                    break
                elif ty==cy and tx==cx: #chaser
                    s[direc_i]=2
                    break
            s[dist_i]=get_dist([ty-dy,tx-dx,ry,rx])
            direc_i+=1;dist_i+=1
    s[20]=get_direc(s,e)
    s[21]=get_dist(s)
    return s

def r_both_pos(state): #generate random pos for both entities, which is not overlapping with themselves nor the walls
    state[:4]=np.random.randint(0,grid_size,4)
    while [state[0],state[1]]==[state[2],state[3]] or [n for n in state[:2]] in walls or [n for n in state[2:4]] in walls:
        state[:4]=np.random.randint(0,grid_size,4)
    return state

def animate_record(num_recs,name):
    #setup model like in train
    chaser_agent=setup(*chaser_setup_param[:4],0,0,1) #start,end epsilon = 0,0 for no random_action
    #runner_agent=setup(*runner_setup_param[:4],0,0,1)
    chaser_agent.load(chaser_agent_name+name)
    #runner_agent.load(runner_agent_name+name)
    record=[]
    #max_step=50
    #steps=0
    for _ in range(num_recs):
        base_state=np.array([4,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        base_state=r_both_pos(base_state)
        chaser_state=update_state(base_state,"chaser")
        #runner_state=update_state(base_state,"runner")
        t_rec=[np.copy(chaser_state)]
        while [chaser_state[0],chaser_state[1]]!=[chaser_state[2],chaser_state[3]]:
            """runner_action=runner_agent.act(runner_state)
            runner_state,r=step(runner_state,runner_action,"runner")
            runner_state=update_state(runner_state,"runner")
            chaser_state[:4]=runner_state[:4]
            t_rec.append(chaser_state)"""

            chaser_action=chaser_agent.act(chaser_state)
            chaser_state,r=step(chaser_state,chaser_action,"chaser")
            chaser_state=update_state(chaser_state,"chaser")
            #runner_state[:4]=chaser_state[:4]
            t_rec.append(np.copy(chaser_state))

            #steps+=1
            #if steps>max_step:break
        chaser_agent.stop_episode()
        #runner_agent.stop_episode()
        record.append(t_rec)
        print(len(t_rec))
        #print(t_rec)
    return record

def animate(record):
    import pygame
    window_x,window_y=500,500
    px=window_x//grid_size
    def draw_rect(c,color="white",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))
    for recs in record:
        while len(recs)>0:
            pygame.time.delay(50)
            window.fill(pygame.Color("black"))
            for n in walls:
                draw_rect([n[1]*px,n[0]*px,px,px],"white")
            a=recs.pop(0)
            draw_rect([a[3]*px,a[2]*px,px,px],"green",0) #runner
            draw_rect([a[1]*px,a[0]*px,px,px],"red",0) #chaser
            pygame.display.flip()
        pygame.time.delay(500)
    pygame.quit()

def train_chaser(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100):
    #agent setup below
    agent=setup(*chaser_setup_param,num_episodes) 
    #loading prexisting agent
    if load:
        agent.load(chaser_agent_name+name)

    longest_dist=get_dist(np.array([0,4,4,0])) #longest possible dist
    max_step=50 #time/step limit
    target_max_step=15
    mem=4 #prev pos record
    timeout_states=[]

    for episode in range(num_episodes): #episode loop
        state=np.array([grid_size,0,0,grid_size,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #cx,cy,rx,ry,cardinal direcs and distance [N.NE.E.SE.S.SW.W.NW , Nd,NEd,Ed,SEd,Sd,SWd,Wd,NWd]

        #choose from either timeout_states, or other option
        if len(timeout_states)>0:
            if np.random.uniform(0,1)>0.5:
                s=r_both_pos(state)
            else:
                #print("-----------------------------------------------")
                state=np.copy(timeout_states.pop(0))
        else:
            s=r_both_pos(state)
        state=update_state(state,"chaser")

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

            state,r=step(state,action,"chaser") #update state'spositions from given action
            state=update_state(state,"chaser") #update the rest of states
            
            d=get_dist(state[:4])
            if d<dist:r+=1
            elif d>dist:r-=1
            dist=d

            total_r+=r #update current episode's total reward
            step_taken+=1 #increment step counter
            #time limit, detect failed catch
            if step_taken>=max_step:
                r-=100 #penalize
                timeout_states.append(np.copy(start_state))
                caught=False
        if caught:
            #print("***********************************************")
            r+=10**6
        total_r+=r
        agent.stop_episode_and_train(state,r,caught) #final train of episode
        t_rec.append(np.copy(state)) #animation
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}, length of timeouts: {len(timeout_states)}")
            print()
        #saving agent models in intervals
        if episode%save_interval==0 and interv_save:
            agent.save(chaser_agent_name+name)
        #periodical aniamtion
        if animate_interval>0 and episode%animate_interval==0:
            animate([t_rec])
    #saving agent models after all episode ran
    if save:
        agent.save(chaser_agent_name+name)
    #display basic values of network
    print(agent.get_statistics())

def train_runner_with_chaser(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100):
    #setup agents
    chaser_agent=setup(*chaser_setup_param,num_episodes)
    runner_agent=setup(*runner_setup_param,num_episodes)
    if load:
        runner_agent.load(runner_agent_name+name)
    chaser_agent.load(chaser_agent_name+name)

    longest_dist=get_dist(np.array([0,4,4,0])) #longest possible dist
    max_step=30 #time/step limit
    mem=4 #prev pos record
    timeout_states=[]
    wins=0

    for episode in range(num_episodes): #episode loop
        base_state=np.array([4,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        base_state=r_both_pos(base_state)
        chaser_state=update_state(np.copy(base_state),"chaser")
        runner_state=update_state(np.copy(base_state),"runner")

        start_state=[n for n in runner_state] #record starting state

        #set vars
        dist=get_dist(runner_state[:4])
        r=0
        total_r=0
        step_taken=0
        caught=True #caught or not
        t_rec=[np.copy(runner_state)] #animate

        if (episode-1)%disp_interval==0:
            wins=0

        while [runner_state[0],runner_state[1]]!=[runner_state[2],runner_state[3]] and caught: #step loop
            
            #runner first
            runner_action=runner_agent.act_and_train(runner_state,r)
            runner_state,r=step(runner_state,runner_action,"runner")
            runner_state=update_state(runner_state,"runner")
            chaser_state[:4]=runner_state[:4] #sync pos
            t_rec.append(np.copy(runner_state)) #animation

            #then chaser
            chaser_action=chaser_agent.act(chaser_state)
            chaser_state,r=step(chaser_state,chaser_action,"chaser")
            chaser_state=update_state(chaser_state,"chaser")
            runner_state[:4]=chaser_state[:4] #sync pos
            t_rec.append(np.copy(runner_state))

            #for surviving
            r+=step_taken/max_step

            total_r+=r #upd cur eps total reward
            step_taken+=1 #step upd
            #step limit reached
            if step_taken>=max_step:
                r+=10
                caught=False
                wins+=1
                #print("***********************************************")
        if caught:
            r-=100
            timeout_states.append(np.copy(start_state))
        total_r+=r
        runner_agent.stop_episode_and_train(runner_state,r,caught) #final train of episode
        chaser_agent.stop_episode()
        if episode%disp_interval==0:
            print(f"episode: {episode}, elapsed episodes: {disp_interval}")
            print(f"wins: {wins}, win rate: {wins*100/disp_interval}%")
            print()
        #saving agent models in intervals
        if episode%save_interval==0 and interv_save:
            runner_agent.save(runner_agent_name+name)
        #periodical aniamtion
        #if animate_interval>0 and episode%animate_interval==0:
        if not caught and episode>=num_episodes//2:
            animate([t_rec])
            pass
    #saving agent models after all episode ran
    if save:
        runner_agent.save(runner_agent_name+name)
    #display basic values of network
    print(runner_agent.get_statistics())
    print(chaser_agent.get_statistics())

def train_both(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100):
    #setup agents
    chaser_agent=setup(*chaser_setup_param,num_episodes)
    runner_agent=setup(*runner_setup_param,num_episodes)
    if load:
        runner_agent.load(runner_agent_name+name)
        chaser_agent.load(chaser_agent_name+name)

    longest_dist=get_dist(np.array([0,grid_size,grid_size,0])) #longest possible dist
    max_step=50 #time/step limit
    target_step=20
    mem=4 #prev pos record
    timeout_states=[]
    wins=0
    total_chaser_r=total_runner_r=0
    total_steps=0
    for episode in range(num_episodes): #episode loop
        max_step-=(max_step-target_step)/num_episodes
        base_state=np.array([grid_size,0,0,grid_size,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        base_state=r_both_pos(base_state)
        chaser_state=update_state(np.copy(base_state),"chaser")
        runner_state=update_state(np.copy(base_state),"runner")

        start_state=[n for n in runner_state] #record starting state

        #set vars
        dist=get_dist(runner_state[:4])
        chaser_r=runner_r=0
        step_taken=0
        caught=True #caught or not
        t_rec=[np.copy(runner_state)] #animate

        if (episode-1)%disp_interval==0:
            wins=0
            total_chaser_r=total_runner_r=0
            total_steps=0

        while [runner_state[0],runner_state[1]]!=[runner_state[2],runner_state[3]] and caught: #step loop
            
            #runner first
            runner_action=runner_agent.act_and_train(runner_state,runner_r)
            runner_state,runner_r=step(runner_state,runner_action,"runner")
            runner_state=update_state(runner_state,"runner")
            chaser_state[:4]=runner_state[:4] #sync pos
            t_rec.append(np.copy(runner_state)) #animation

            #then chaser
            chaser_action=chaser_agent.act_and_train(chaser_state,chaser_r)
            chaser_state,chaser_r=step(chaser_state,chaser_action,"chaser")
            chaser_state=update_state(chaser_state,"chaser")
            runner_state[:4]=chaser_state[:4] #sync pos
            t_rec.append(np.copy(runner_state))

            #for runner
            #runner_r+=step_taken

            #for chaser
            d=get_dist(chaser_state[:4])
            if d<dist:
                chaser_r+=1
                #runner_r-=1
            elif d>dist:
                chaser_r-=1
                #runner_r+=1
            dist=d

            total_chaser_r+=chaser_r
            total_runner_r+=runner_r
            step_taken+=1 #step upd

            #if runner not caught for max_step
            if step_taken>=max_step:
                runner_r+=100000
                chaser_r-=100000
                caught=False
                wins+=1
        #if chaser catches runner
        if caught:
            runner_r-=100000
            chaser_r+=100000
            timeout_states.append(np.copy(start_state))
        #final summing of reward of ep
        total_chaser_r+=chaser_r
        total_runner_r+=runner_r
        total_steps+=step_taken
        #final train of ep
        runner_agent.stop_episode_and_train(runner_state,runner_r,caught)
        chaser_agent.stop_episode_and_train(chaser_state,chaser_r,caught)
        if episode%disp_interval==0:
            print(f"episode: {episode}, elapsed episodes: {disp_interval}, step limit: {max_step}, avg steps: {total_steps/disp_interval}")
            print(f"runner wins: {wins}, runner win rate: {wins*100/disp_interval}%, avg chaser reward: {total_chaser_r/disp_interval}, avg runner reward: {total_runner_r/disp_interval}")
            print()
        #saving agent models in intervals
        if episode%save_interval==0 and interv_save:
            runner_agent.save(runner_agent_name+name)
            chaser_agent.save(chaser_agent_name+name)
        #periodical aniamtion
        if animate_interval>0 and episode%animate_interval==0:
            animate([t_rec])
    #saving agent models after all episode ran
    if save:
        runner_agent.save(runner_agent_name+name)
        chaser_agent.save(chaser_agent_name+name)
    #display basic values of network
    print(runner_agent.get_statistics())
    print(chaser_agent.get_statistics())

train_chaser(5001,"1",True,False,True,1,200) #(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,animate_interval=-1,save_interval=100)
#animate(animate_record(50,"1"))

#main_chaser1 well trained navigation bot