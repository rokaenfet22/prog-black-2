import pygame
import time
from game.Player import Runner,Catcher
from dqn_tools import *

def train():
    SAVE_PATH = 'saved_models/tag1/navigation'
    LOAD_REPLAY_BUFFER = True

    USE_PER=True                     # Use Priority Experience Replay
    PRIORITY_SCALE=0.5               #How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
    #hyper
    TOTAL_FRAMES = 300000          #Total number of frames to train for
    MAX_EPISODE_LENGTH = 50       # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
    FRAMES_BETWEEN_EVAL = 5000     # Number of frames between evaluations
    EVAL_LENGTH = 500              # Number of frames to evaluate for
    eps_annealing_frames=75000

    DISCOUNT_FACTOR = 0.3            # Gamma, how much to discount future rewards
    MIN_REPLAY_BUFFER_SIZE = 400    # The minimum size the replay buffer must be before we start to update the agent
    MEM_SIZE = 1000000                # The maximum size of the replay buffer

    UPDATE_FREQ = 2                   # Number of actions between gradient descent steps
    TARGET_UPDATE_FREQ = 50         # Number of actions between when the target network is updated

    INPUT_SHAPE = (8,)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
    screen_size = (5, 5)
    BATCH_SIZE = 4               # Number of samples the agent learns from at once
    LEARNING_RATE = 0.001

    #ENV details

    init_catcher_pos=(1,1)
    init_runner_pos=(4,4)
    max_walls=7
    player_size=1
    wall_thickness=1
    walls=[[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667], [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332], [0, 25.0, 16.666666666666664, 2.666666666666667], [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667], [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334], [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667], [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
    player_a=1
    pygame.init()
    # Set up the display
    pygame.display.set_caption("Tag")
    screen = pygame.display.set_mode((screen_size[0], screen_size[1]))
    #set up the players
    catcher=Catcher(init_catcher_pos[0],init_catcher_pos[1],[255,0,0],player_size)
    runner=Runner(init_runner_pos[0],init_runner_pos[1],[0,0,255],player_size)
    #init tf_agents environment

    game_wrapper=GameWrapper(catcher,runner,wall_list=[],screen_size=screen_size,acceleration=player_a,screen=screen,no_op_steps=MAX_NOOP_STEPS,init_catcher_pos=init_catcher_pos,init_runner_pos=init_runner_pos)

    # Build main and target networks
    MAIN_DQN = build_q_network(game_wrapper.action_space.n, learning_rate=LEARNING_RATE, input_shape=INPUT_SHAPE,screen_size=screen_size)

    TARGET_DQN = build_q_network(game_wrapper.action_space.n, learning_rate=LEARNING_RATE,input_shape=INPUT_SHAPE,screen_size=screen_size)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE,use_per=USE_PER)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE,use_per=USE_PER,replay_buffer_start_size=MIN_REPLAY_BUFFER_SIZE,eps_annealing_frames=eps_annealing_frames)
    #agent.load('saved_models/tag1/navigation/save-00005049/')

    # TRAINING
    frame_number = 0
    rewards = []
    loss_list = []

    while frame_number < TOTAL_FRAMES:
          # Training
          epoch_frame = 0
          while epoch_frame < FRAMES_BETWEEN_EVAL:
              start_time = time.time()
              game_wrapper.reset(rand=True)
              episode_reward_sum = 0
              for i in range(MAX_EPISODE_LENGTH):
                  # Get action
                  state = np.array(game_wrapper.get_state())
                  #game_wrapper.render()
                  state = np.reshape(state, (1, 8))
                  action = agent.get_action(frame_number, state)
                  # Take step
                  new_state, reward, terminal, info = game_wrapper.step(action)
                  new_state=np.array(new_state)
                  #reward=reward - i*STALLING_PENALTY #penalty proportional to the current episode frame number
                  frame_number += 1
                  epoch_frame += 1
                  episode_reward_sum += reward
                  # Add experience to replay memory
                  agent.add_experience(action=action,
                                      frame=new_state,
                                      reward=reward,
                                      terminal=terminal)

                  # Update agent
                  if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                      loss, _ = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,priority_scale=PRIORITY_SCALE)

                  # Update target network
                  if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                      agent.update_target_network()

                  # Break the loop when the game is over
                  if terminal:
                      print("terminal")
                      terminal = False
                      break

              rewards.append(episode_reward_sum)
              # Output the progress every 10 games
              if len(rewards) % 10 == 0:
                  print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

          # Save model
          print("model saved")
          agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}')

          # Evaluation every `FRAMES_BETWEEN_EVAL` frames
          terminal = True
          eval_rewards = []
          evaluate_frame_number = 0
          for _ in range(EVAL_LENGTH):
              if terminal:
                  game_wrapper.reset(rand=True)
                  episode_reward_sum = 0
                  episode_frame_number = 0
                  terminal=False
              state=np.reshape(np.array(game_wrapper.get_state()),(1,8))
              action=agent.get_action(frame_number, state, evaluation=True)
              # Step action
              _, reward, terminal, info = game_wrapper.step(action)
              episode_frame_number+=1
              #reward=reward-episode_frame_number*STALLING_PENALTY
              evaluate_frame_number += 1
              episode_reward_sum += reward

              # On game-over
              if terminal:
                  eval_rewards.append(episode_reward_sum)
                  game_wrapper.reset(rand=True)


          if len(eval_rewards) > 0:
              final_score = np.mean(eval_rewards)
          else:
              # In case the game is longer than the number of frames allowed
              final_score = episode_reward_sum
          # Print score and write to tensorboard
          print('Evaluation score:', final_score)

train()