import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

"""
states = (Aoff,Boff),(Aon,Boff),(Aoff,Bon),(Aon,Bon) = 0,1,2,3
actions = Pressing A,B,C = 0,1,2
reward = 1 when st=3 and a=2 otherwise 0
"""

sa_table=[[1,2,0],[0,3,1],[3,0,2],[2,1,3]]

class QFunction(chainer.Chain):
    def __init__(self,obs_size,n_actions,n_hidden_channels=2):
        super(QFunction,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False):
        h1=F.tanh(self.l1(x))
        h2=F.tanh(self.l2(h1))
        y=chainerrl.action_value.DiscreteActionValue(self.l3(h2))
        return y
    
def random_action():
    return np.random.choice([0,1,2])

def step(s,a):
    if s[0]==3 and a==2: r=1
    else: r=0
    return np.array([sa_table[s[0]][a]]),r

def train():
    gamma=0.9
    max_number_of_steps=10
    num_episodes=500

    q_func=QFunction(1,3)
    optimizer=chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1,end_epsilon=0.1,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10**6)
    phi=lambda x: x.astype(np.float32,copy=False)
    agent=chainerrl.agents.DoubleDQN(q_func,optimizer,replay_buffer,gamma,explorer,replay_start_size=500,update_interval=1,target_update_interval=200,phi=phi)

    for episode in range(num_episodes):
        state=np.array([0])
        R=0
        reward=0
        done=True
        for t in range(max_number_of_steps):
            action=agent.act_and_train(state,reward)
            state,reward=step(state,action)
            #print(state,action,reward)
            R+=reward #add reward
        agent.stop_episode_and_train(state,reward,done)
        if episode%10==0:
            print(f"episode: {episode+1} total reward: {R}")
    agent.save("mouse_agent")
    print(agent.get_statistics())

def test():
    gamma=0.9
    max_number_of_steps=10
    num_episodes=3000

    q_func=QFunction(1,3)
    optimizer=chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1,end_epsilon=0.1,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
    phi=lambda x: x.astype(np.float32,copy=False)
    agent=chainerrl.agents.DQN(q_func,optimizer,replay_buffer,gamma,explorer,replay_start_size=500,update_interval=1,target_update_interval=200,phi=phi)
    agent.load("mouse_agent")

    total=0
    ep_num=50
    for episode in range(ep_num):
        state=np.array([np.random.choice([0,1,2,3])])
        #state=np.array([0])
        R=0
        reward=0
        done=True
        for t in range(max_number_of_steps):
            #action=agent.act_and_train(state,reward)
            action=agent.act(state)
            state,reward=step(state,action)
            #print(f"state={state}, action={action}, reward={reward}")
            R+=reward
        agent.stop_episode()
        if episode%1==0:
            print(f"episode: {episode+1} total reward: {R}")
            print()
            total+=R

        print(f"average = {total/ep_num}")

train()
test()