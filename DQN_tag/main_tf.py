import pygame
import time
from game.Player import Runner,Catcher
from dqn_tools import *
#walls, no momentum. Random runner and catcher start positions 5x5. Navigation. without adding all wall info in states
def train():
    SAVE_PATH = 'saved_models/tag2/navigation/walls_no_v'
    LOAD_REPLAY_BUFFER = True

    USE_PER=True                    # Use Priority Experience Replay
    PRIORITY_SCALE=0.7               #How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
    #hyper
    TOTAL_FRAMES = 25000          #Total number of frames to train for
    MAX_EPISODE_LENGTH = 40       # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
    FRAMES_BETWEEN_EVAL = 5000     # Number of frames between evaluations
    EVAL_LENGTH = 5              # Number of games to evaluate for
    eps_annealing_frames=25000

    DISCOUNT_FACTOR = 0.7            # Gamma, how much to discount future rewards
    MIN_REPLAY_BUFFER_SIZE = 300    # The minimum size the replay buffer must be before we start to update the agent
    MEM_SIZE = 1000000                # The maximum size of the replay buffer

    UPDATE_FREQ = 2                   # Number of actions between gradient descent steps
    TARGET_UPDATE_FREQ = 25         # Number of actions between when the target network is updated

    INPUT_SHAPE = (28,)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
    screen_size = (5, 5)
    BATCH_SIZE = 32              # Number of samples the agent learns from at once
    LEARNING_RATE = 0.001
    layer_structure=(70,10,5)
    #ENV details

    init_catcher_pos=(1,1)
    init_runner_pos=(4,4)
    render_scale_factor = 100  # scale everything up when rendering on screen for better visibilty

    player_size=1
    walls1 = np.array([[1, 0, 1, 2], [3, 2, 2, 1]])  # constructed on a 5x5 grid
    walls = (walls1 / 5) * screen_size[0]  # scale the walls according to screen size
    player_a=1
    pygame.init()
    # Set up the display
    pygame.display.set_caption("Tag")
    screen = pygame.display.set_mode((screen_size[0]*render_scale_factor, screen_size[1]*render_scale_factor))
    #set up the players
    catcher=Catcher(init_catcher_pos[0],init_catcher_pos[1],[255,0,0],player_size)
    runner=Runner(init_runner_pos[0],init_runner_pos[1],[0,0,255],player_size)
    #init tf_agents environment

    game_wrapper=GameWrapper(catcher,runner,wall_list=walls,screen_size=screen_size,acceleration=player_a,screen=screen,init_catcher_pos=init_catcher_pos,init_runner_pos=init_runner_pos)

    # Build main and target networks
    MAIN_DQN = build_q_network(layer_structure=layer_structure,n_actions=game_wrapper.action_space.n, learning_rate=LEARNING_RATE, input_shape=INPUT_SHAPE,screen_size=screen_size)

    TARGET_DQN = build_q_network(layer_structure=layer_structure,n_actions=game_wrapper.action_space.n, learning_rate=LEARNING_RATE,input_shape=INPUT_SHAPE,screen_size=screen_size)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE,use_per=USE_PER)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.action_space.n, input_shape=INPUT_SHAPE,max_frames=TOTAL_FRAMES, batch_size=BATCH_SIZE,use_per=USE_PER,replay_buffer_start_size=MIN_REPLAY_BUFFER_SIZE,eps_annealing_frames=eps_annealing_frames,eps_initial=1)
    #agent.load('saved_models/tag2/navigation/save-00025059/')

    # TRAINING
    frame_number = 0
    rewards = []
    while frame_number < TOTAL_FRAMES:
          # Training
          epoch_frame = 0
          while epoch_frame < FRAMES_BETWEEN_EVAL:
              start_time = time.time()
              game_wrapper.reset(rand=True)
              episode_reward_sum = 0


              for i in range(MAX_EPISODE_LENGTH):
                  # Get action
                  state = np.array(game_wrapper.get_state(reduced=False))

                  #game_wrapper.render(len_scale_factor=render_scale_factor)
                  state = np.reshape(state, (1, INPUT_SHAPE[0]))

                  action = agent.get_action(frame_number, state)
                  # Take step
                  new_state, reward, terminal, info = game_wrapper.step(action)
                  new_state=np.array(new_state)
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
                      loss, _ = agent.learn(batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=replay_buffer.count,priority_scale=PRIORITY_SCALE)

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
                  print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(replay_buffer.count).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

          # Save model
          print("model saved")
          agent.save(f'{SAVE_PATH}/save-{str(replay_buffer.count).zfill(8)}')

          # Evaluation every `FRAMES_BETWEEN_EVAL` frames
          terminal = True
          eval_rewards = []
          wins=0
          for game in range(EVAL_LENGTH):
              game_wrapper.reset(rand=True)
              game_wrapper.render(len_scale_factor=render_scale_factor)
              time.sleep(0.1)
              episode_reward_sum = 0
              if not terminal:
                  eval_rewards.append(0)
              for frame in range(MAX_EPISODE_LENGTH+20):
                  state=np.reshape(np.array(game_wrapper.get_state(reduced=False)),(1,INPUT_SHAPE[0]))
                  action=agent.get_action(frame_number, state, evaluation=True)
                  # Step action
                  _, reward, terminal, info = game_wrapper.step(action)
                  episode_reward_sum += reward
                  game_wrapper.render(len_scale_factor=render_scale_factor)
                  time.sleep(0.1)
                  # On game-over
                  if terminal:
                      wins+=1
                      eval_rewards.append(episode_reward_sum)
                      print(f'Game over, reward: {episode_reward_sum}')
                      break

          print(f'wins:{wins}; losses:{EVAL_LENGTH - wins}')
          print('Average reward:', np.mean(eval_rewards))

train()