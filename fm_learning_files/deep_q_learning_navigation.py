import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

sa_table=[
[[1],[0],[0],[5]], [[2],[0],[1],[1]], [[3],[1],[2],[2]], [[4],[2],[3],[3]], [[4],[3],[4],[9]],
[[5],[5],[0],[10]], [[7],[5],[1],[11]], [[8],[6],[2],[12]], [[9],[7],[3],[13]], [[9],[9],[4],[14]],
[[10],[10],[5],[15]], [[12],[10],[6],[16]], [[12],[12],[12],[12]], [[14],[12],[8],[18]], [[14],[15],[9],[19]],
[[15],[15],[10],[20]], [[17],[15],[11],[21]], [[18],[16],[12],[22]], [[19],[17],[13],[23]], [[19],[19],[14],[24]],
[[21],[20],[15],[20]], [[22],[20],[21],[21]], [[23],[21],[22],[22]], [[24],[22],[23],[23]], [[24],[23],[19],[24]]
]

class QFunction(chainer.Chain):
    def __init__(self,obs_size,n_actions,n_hidden_channels=50):
        super(QFunction,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False):
        h1=F.relu(self.l1(x))
        h2=F.relu(self.l2(h1))
        y=chainerrl.action_value.DiscreteActionValue(self.l3(h2))
        return y

def random_action():
    return np.random.choice([0,1,2,3])

def step(s,a):
    r=0
    if sa_table[s[0]][a][0]==15: r=-2000
    elif sa_table[s[0]][a][0]==20: r=1000
    return np.array([sa_table[s[0]][a][0]]),r

gamma=0.9
alpha=0.5
max_number_of_steps=500
num_episodes=100

q_func=QFunction(1,4) #25 states 4 actions
optimizer=chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0,end_epsilon=0.1,decay_steps=num_episodes//100,random_action_func=random_action)
replay_buffer=chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
phi=lambda x: x.astype(np.float32,copy=False)
agent=chainerrl.agents.DoubleDQN(q_func,optimizer,replay_buffer,gamma,explorer,replay_start_size=500,update_interval=1,target_update_interval=100,phi=phi)

agent.load("agent")

for episode in range(num_episodes):
    state=np.array([0])
    reward=0
    R=0
    step_taken=0
    done=True
    next_state=[0]
    while state[0]!=20 : #one try's iteration and step_taken<=max_number_of_steps
        action=agent.act_and_train(state,reward)
        next_state,reward=step(state,action)
        R+=reward
        #print(f"state={state+1}, action={action}, reward={reward}")
        state=next_state
        step_taken+=1 #count the number steps needed to reach goal
    agent.stop_episode_and_train(state,R,done)
    print(f"episode: {episode+1}, total reward: {R}")
    print(f"steps taken: {step_taken}")
    print()
    if episode%50==0:
        agent.save("agent")