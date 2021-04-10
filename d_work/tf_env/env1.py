import pygame

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec,tensor_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

screen_size=(50,50)
init_it_pos=(16.7,16.7)
init_player_pos=(25,25)
max_walls=7


class TagEnv(py_environment.PyEnvironment):
    def __init__(self,it_player,player,wall_list,screen_size,acceleration,screen):
        self.it_player = it_player
        self.a=acceleration
        self.player = player
        self.wall_list = wall_list
        self.screen_w,self.screen_h=screen_size
        self.screen=screen
        self.steps=0

        # There are 5 actions, corresponding to arrow keys with 0 being pressing nothing
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        '''
        more complex observation
        observation space=
        [wall1,
        wall2,
        wall3,
        .
        .
        .
        wall7,
        [it_x,it_y,opp_x,opp_y],
        [screen_width,screen_height,it_vx,it_vy],
        [opp_vx,opp_vy,opp_a,it_a]
        
        ]
        '''

        '''
        simpler, assumes the walls and acceleration configs are constant. Will not be able to adapt to new walls and a
        [
        [it_x,it_y,opp_x,opp_y,it_vx,it_vy,opp_vx,opp_vy]
        ]
        '''
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8,), dtype=np.float32, minimum=[0,0,0,0,-5,-5,-5,-5], maximum=[screen_size[0],screen_size[0],screen_size[0],screen_size[0],5,5,5,5], name='observation')
        self._state=[init_it_pos[0],init_it_pos[1],init_player_pos[0],init_player_pos[1],0,0,0,0]
        self._current_time_step = ts.restart(np.array(self._state, dtype=np.float32))

        self._episode_ended=False

    def observation_spec(self):
        return self._observation_spec
    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        """Describes the `TimeStep` fields returned by `step()`.
        Override this method to define an environment that uses non-standard values
        for any of the items returned by `step()`. For example, an environment with
        array-valued rewards.
        Returns:
          A `TimeStep` namedtuple containing (possibly nested) `ArraySpec`s defining
          the step_type, reward, discount, and observation structure.
        """
        return ts.time_step_spec(tensor_spec.from_spec(self.observation_spec()), tensor_spec.from_spec(self.reward_spec()))

    def _reset(self):
        self._state = [init_it_pos[0],init_it_pos[1],init_player_pos[0],init_player_pos[1],0,0,0,0]

        self.steps=0
        self._episode_ended = False
        self._current_time_step=ts.restart(np.array(self._state, dtype=np.float32))
        return self._current_time_step
    def _step(self,action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self._reset()

        if action == 0:  # UP
            self.it_player.accelerate(0, -self.a)
        elif action == 1:  # RIGHT
            self.it_player.accelerate(self.a, 0)
        elif action == 2:  # DOWN
            self.it_player.accelerate(0, self.a)
        elif action == 3:  # LEFT
            self.it_player.accelerate(-self.a, 0)

        self.it_player.move(self.screen_size, self.wall_list)

        self.steps+=1
        if self.it_player.rect.colliderect(self.player):
            self._episode_ended=True
        else:
            self._episode_ended=False
        self._state=self.get_state()
        if self._episode_ended:
            reward = 75+np.sqrt(self.steps)
            return ts.termination(np.array(self._state,dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self._state,dtype=np.float32),reward=self.calculate_reward(), discount=1.0)

    def calculate_reward(self):
        a = np.array(self.it_player.get_pos())
        b = np.array(self.player.get_pos())
        return float(75-(np.sum((a-b)**2))**(1/2))
    def get_state(self):
        '''
        [
        [it_x,it_y,opp_x,opp_y,it_vx,it_vy,opp_vx,opp_vy]
        ]
        '''
        state=[self.it_player.get_pos()[0],self.it_player.get_pos()[1],self.player.get_pos()[0],self.player.get_pos()[1],self.it_player.get_velocity()[0],self.it_player.get_velocity()[1],self.player.get_velocity()[0],self.player.get_velocity()[1]]
        return state
    def get_init_state(self):
        state=[
            [init_it_pos[0], init_it_pos[1], init_player_pos[0], init_player_pos[1], 0, 0, 0, 0]
        ]
        return state
    def render(self):
        # Draw the scene
        self.screen.fill((255, 255, 255))
        # fill screen excess black
        if self.screen.get_size() != (self.screen_w,self.screen_h):
            r = pygame.Rect(self.screen_w, 0, self.screen.get_size()[0] - self.screen_w,
                            self.screen_h)
            pygame.draw.rect(self.screen, (0, 0, 0), r)
        for wall in self.wall_list:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            pygame.draw.rect(self.screen, (0, 0, 0), wall_rect)
        pygame.draw.rect(self.screen, self.it_player.color, self.it_player.rect)
        pygame.draw.rect(self.screen, self.player.color, self.player.rect)
        pygame.display.flip()




