from models.DQN_tf.dqn_tools import GameWrapper,build_q_network,ReplayBuffer,Agent
import numpy as np
import pygame
import tensorflow as tf
from game.Player import Runner,Catcher
import time

#EVALUATION
# hyper
EVAL_LENGTH = 900  # Number of frames to evaluate for

MEM_SIZE = 1000000  # The maximum size of the replay buffer

MAX_NOOP_STEPS = 1  # Randomly perform this number of actions before every evaluation to give it an element of randomness
eps_annealing_frames = 100000
use_per=True
INPUT_SHAPE = (8,)  # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 1  # Number of samples the agent learns from at once
LEARNING_RATE = 0.001
screen_size = (5, 5)
render_scale_factor=100   #scale everything up when rendering on screen for better visibilty
MIN_REPLAY_BUFFER_SIZE = 400  # The minimum size the replay buffer must be before we start to update the agent

# ENV details
init_catcher_pos = (1, 1)
init_runner_pos = (4, 4)
max_walls = 7 #need in order for the full state to have consistent shape
player_size = 1
wall_thickness = 1
walls = [[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667],
         [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332],
         [0, 25.0, 16.666666666666664, 2.666666666666667],
         [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667],
         [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334],
         [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667],
         [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
player_a = 1
pygame.init()
# Set up the display
pygame.display.set_caption("Tag")
screen = pygame.display.set_mode((screen_size[0]*render_scale_factor, screen_size[1]*render_scale_factor))

# set up the players
catcher = Catcher(init_catcher_pos[0], init_catcher_pos[1], [255, 0, 0], player_size)
runner = Runner(init_runner_pos[0], init_runner_pos[1], [0, 0, 255], player_size)
# init tf_agents environment
game_wrapper = GameWrapper(catcher, runner, wall_list=[], screen_size=screen_size, acceleration=player_a, screen=screen,
                           no_op_steps=MAX_NOOP_STEPS, init_catcher_pos=init_catcher_pos, init_runner_pos=init_runner_pos)

# Build main and target networks
MAIN_DQN = build_q_network(game_wrapper.action_space.n, learning_rate=LEARNING_RATE,screen_size=screen_size, input_shape=INPUT_SHAPE)
# for layer in MAIN_DQN.layers:
#     print(layer, layer.input_shape)
TARGET_DQN = build_q_network(game_wrapper.action_space.n,learning_rate=LEARNING_RATE,screen_size=screen_size ,input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.action_space.n, input_shape=INPUT_SHAPE,
              batch_size=BATCH_SIZE,use_per=use_per,eps_annealing_frames=eps_annealing_frames,replay_buffer_start_size=MIN_REPLAY_BUFFER_SIZE)
model_path='saved_models/tag1/navigation/save-00100374'
print('Loading model...')
agent.load(model_path)
print('Loaded')

terminal = True
eval_rewards = []
evaluate_frame_number = 0
episode_reward_sum=0
game_wrapper.reset(rand=True)
game_wrapper.render(len_scale_factor=render_scale_factor)
for frame in range(EVAL_LENGTH):

    state=np.array(game_wrapper.get_state())
    state=np.reshape(state,(1,8))
    print(state)
    action=agent.get_action(0, state, evaluation=True)

    # Step action
    new_state, reward, terminal, info = game_wrapper.step(action)
    print(reward)
    game_wrapper.render(len_scale_factor=render_scale_factor)
    evaluate_frame_number += 1
    episode_reward_sum += reward
    time.sleep(0.1)

    # On game-over
    if terminal: #if the catcher caught the runner
        print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)
        game_wrapper.reset(rand=True)
        game_wrapper.render(len_scale_factor=render_scale_factor)
        episode_reward_sum = 0
        print("terminal")
        terminal = False


print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)