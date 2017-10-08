# Keras implementation, will return about the following:

# Using TensorFlow backend.
# [2017-10-07 19:33:50,746] Making new env: CartPole-v0
# Episode 0, last-100-avg-reward: 30, epsilon: 0.70, seconds: 0.38
# Episode 10, last-100-avg-reward: 14, epsilon: 0.60, seconds: 0.57
# Episode 20, last-100-avg-reward: 14, epsilon: 0.52, seconds: 0.76
# ...
# Episode 100, last-100-avg-reward: 64, epsilon: 0.15, seconds: 3.92
# ...
# Episode 250, last-100-avg-reward: 145, epsilon: 0.10, seconds: 14.07
# Episode 260, last-100-avg-reward: 154, epsilon: 0.10, seconds: 14.92
# Episode 270, last-100-avg-reward: 161, epsilon: 0.10, seconds: 15.78
# Episode 280, last-100-avg-reward: 170, epsilon: 0.10, seconds: 16.61
# Episode 290, last-100-avg-reward: 183, epsilon: 0.10, seconds: 17.43
# Episode 300, last-100-avg-reward: 194, epsilon: 0.10, seconds: 18.26
# SOLVED AT EPISODE 301, time: 18.36s



import gym
import keras
import numpy as np
import random
import signal
import collections
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

class KeyboardCtrlC:
    def __init__(self):
        self.key_pressed = False
        signal.signal(signal.SIGINT, self.key_pressed_m)
        signal.signal(signal.SIGTERM, self.key_pressed_m)

    def key_pressed_m(self, signum, frame):
        self.key_pressed = True


def get_model(obs=4, learning_rate=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, input_shape=(obs,), activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(2, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='mse',
        metrics=[],
    )

    return model


def main(max_episodes, print_log_episodes=20):
    env = gym.make('CartPole-v0')

    keyboard_input = KeyboardCtrlC()
    transitions = collections.deque(maxlen=1000)
    episodes = collections.deque(maxlen=100)
    start_time = time.time()

    model = get_model()

    for episode in range(max_episodes):
        observation = env.reset()

        for iteration in range(300):
            old_observation = observation

            epsilon = max(0.1, 0.7 * pow(0.985, episode))

            if np.random.random() < epsilon:
                action = np.random.choice(range(2))
            else:
                # action is integer, either 0 or 1
                action = np.argmax(model.predict(np.expand_dims(old_observation, 0)))

            observation, reward, done, info = env.step(action)

            if done and iteration < 199:
                reward = -100 # we failed, punish when game ends too early

            transitions.append([old_observation, action, reward, observation])

            if done:
                break

            if keyboard_input.key_pressed:
                print("Started python console. Quit with 'ctrl-d' or continue with 'c'")
                import ipdb; ipdb.set_trace()

        episodes.append(iteration)
        if episode % print_log_episodes == 0:
            print('Episode {}, last-100-avg-reward: {}, epsilon: {:.2f}, seconds: {:.2f}'.format(episode, sum(episodes)//len(episodes), epsilon, time.time()-start_time))
        if sum(episodes)/len(episodes) > 195:
            print('SOLVED AT EPISODE {}, time: {:.2f}s'.format(episode, time.time()-start_time))
            return episode

        minibatch_size = 128
        if len(transitions) >= minibatch_size:
            sample_transitions = random.sample(transitions, minibatch_size)
            sample_transitions = np.array(sample_transitions)

            train_x = np.array([np.array(x) for x in sample_transitions[:,0]])
            train_y = model.predict(train_x)

            next_state_x = np.array([np.array(x) for x in sample_transitions[:,3]])
            next_state_value_y = model.predict(next_state_x)

            for idx, arr in enumerate(sample_transitions):
                state, action, reward, next_state = arr

                if reward > 0: # normal round, add discounted future reward
                    gamma = 0.99
                    reward += gamma * np.max(next_state_value_y[idx])

                train_y[idx, action] = reward

            model.fit(np.array(train_x), np.array(train_y), epochs=1, verbose=0)

if __name__ == "__main__":
    results = []
    main(600)
    print("Keras done")

