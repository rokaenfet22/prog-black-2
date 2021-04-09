import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from game.Player import Player,IT
from gym_envs.gym_env import TagEnv
import gym
import gym_envs

import pygame
import time
import numpy as np
import random
screen_size=(20,20)
init_it_pos=(3,3)
init_player_pos=(15,15)
max_walls=7


#game parameters
player_size=2
wall_thickness=2
walls=[[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667], [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332], [0, 25.0, 16.666666666666664, 2.666666666666667], [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667], [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334], [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667], [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
player_a=1
#set up the player objects
it_player=IT(init_it_pos[0],init_it_pos[1],[255,0,0],player_size)
player1=Player(init_player_pos[0],init_player_pos[1],[0,0,255],player_size)
pygame.init()
# Set up the display
pygame.display.set_caption("Tag")
screen = pygame.display.set_mode((screen_size[0], screen_size[1]))
#init gym environment
env = TagEnv(it_player,player1,[],screen_size,acceleration=player_a,screen=screen,init_player_pos=init_player_pos,init_it_pos=init_it_pos)


#hyperparameters

def flatten_state(x,y):
    return 21*y+x
#training
def train(num_ep,max_steps_per_ep,learning_rate,gamma,epsilon,max_epsilon,min_epsilon,epsilon_decay,q_table=[]):
    rewards=[]
    if q_table==[]:
        #print("asdasd")
        q_table=np.zeros((441,4))
    for ep in range(num_ep):
        state=env.reset()
        state=flatten_state(state[0],state[1])
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps_per_ep):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)

            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(q_table[state, :])

            # Else doing a random choice --> exploration
            else:
                action = env.action_space.sample()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
            #print( new_state, reward, done, info)
            new_state=flatten_state(new_state[0],new_state[1])
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            q_table[state, action] = q_table[state, action] + learning_rate * (
                        reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

            total_rewards += reward

            # Our new state is state
            state = new_state

            if done == True:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep)
        rewards.append(total_rewards)
    return q_table,sum(rewards)



def test(q_table):
    game_rewards=[]
    current_game_reward=0
    for episode in range(100):
        state = env.reset()
        state=flatten_state(state[0],state[1])
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps_per_ep):
            for _ in range(noop_steps):
                action=random.randint(0,3)
                state,reward, done, info=env.step(action)

            env.render()
            time.sleep(0.2)
            state = flatten_state(state[0], state[1])

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(q_table[state, :])

            new_state, reward, done, info = env.step(action)
            new_state=flatten_state(new_state[0],new_state[1])
            current_game_reward+=reward
            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)

                # We print the number of step it took.
                print("Number of steps", step)
                game_rewards.append(current_game_reward)
                current_game_reward=0
                break
            state = new_state
    env.close()
    print(f'average reward per game:{sum(game_rewards)/100}')

def load_q_table(file):
    f = open(file, "r")
    q_table=[]
    for x in f:
        if x != "/n":
            q_table.append([float(a) for a in x.strip().split(",")])
    f.close()
    return np.array(q_table)

def save_q_table(q_table,file):
    with open(file, "w") as f:
        f.writelines([",".join(list(map(lambda x: str(x), list(r)))) + "\n" for r in q_table])


#hyperparameters
noop_steps=4
num_ep=100000
max_steps_per_ep=250
learning_rate=0.15
gamma=0.99
epsilon=1
max_epsilon=1
min_epsilon=0.01
epsilon_decay=0.001

# q_table=load_q_table("q_table1.txt")
#q_table,reward=train(num_ep,max_steps_per_ep,learning_rate,gamma,epsilon,max_epsilon,min_epsilon,epsilon_decay)
# print(reward)
#save_q_table(q_table,"q_table20x20_nowalls.txt")
q_table=load_q_table("q_table20x20_nowalls.txt")
test(q_table)

# observation = env.reset()
# for t in range(100):
#         env.render()
#         time.sleep(0.01)
#         action = env.action_space.sample()
#
#         observation, reward, done, info = env.step(action)
#         print (observation, reward, done, info)
#         if done:
#             print("Finished after {} timesteps".format(t+1))
#             break
