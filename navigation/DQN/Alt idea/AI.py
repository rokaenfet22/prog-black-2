import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import sys

'''gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
'''
count = 0
#######################################
##### needs fixing: slow learning #####
#######################################

tf.compat.v1.disable_eager_execution()
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    '''model = keras.Sequential()
    model.add(keras.Input(shape=(4,100,100)))
    model.add(keras.layers.Dense(fc1_dims, activation='relu'))
    model.add(keras.layers.Dense(fc2_dims, activation='relu'))
    model.add(keras.layers.Dense((n_actions), activation='relu', name="worstnet"))
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model'''
    Tag_shape = (4, 60, 60, 1)
    frames_input = keras.layers.Input(Tag_shape, name='frames')
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.Conv2D(32, (8, 8), strides=(4,4), activation='relu')(normalized)
    conv_2 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_3 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv_2)
    conv_flattened = keras.layers.Flatten()(conv_3)
    hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
    hidden2 = keras.layers.Dense(512, activation='relu')(hidden)
    output = keras.layers.Dense(n_actions)(hidden2)

    model = keras.Model(inputs=frames_input, outputs=output)
    optimizer=Adam(learning_rate=lr)
    model.compile(optimizer, loss=tf.keras.losses.Huber())

    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-4, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
        self.targetNet = tf.keras.models.clone_model(
                self.q_eval, input_tensors=None, clone_function=None
            )
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if observation.shape == (4,60,60, 3):
            observation = np.array([observation])
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            observation = observation.reshape(1, 4, 60, 60, 1)
            actions = self.q_eval.predict(observation)
            #action = np.argmax(actions)
            try:
                action = np.random.choice(actions[0], 1, p=softmax(actions[0]))
            except ValueError:
                action = np.argmax(actions)
            listy = actions[0]
            action = np.where(listy==action)[0][0]
        return action

    def learn(self):
        global count
        if self.memory.mem_cntr < self.batch_size:
            return

        if count%1000 == 0:
            self.targetNet.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states = states.reshape((32, 4, 60, 60, 1))
        q_eval = self.q_eval.predict(states)
        states_ = states_.reshape((32, 4, 60, 60, 1))
        #self.targetNet.predict(states_) 
        q_next = self.targetNet.predict(states_)
        
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        val = rewards + self.gamma * np.max(q_next, axis=1)*dones
        q_target[batch_index, actions] = val

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        count  += 1
        '''
        states = states.reshape((64, 4, 60, 60, 1))
        states_ = states_.reshape((64, 4, 60, 60, 1))
        action_list = []
        for action in actions:
            action_list.append(np.full(5, action))
        actions = np.array(action_list)
        mask = np.ones(actions.shape)
        next_Q_values = self.q_eval.predict([states_, mask])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[dones] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.q_eval.fit(
            [states, actions], actions * Q_values[:, None],
            epochs=1, batch_size=len(states), verbose=0
        )'''

    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)

'''def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = keras.backend.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term'''

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()