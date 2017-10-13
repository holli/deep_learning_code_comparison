# Keras TensorFlow implementation, run with "python keras_tf.py"

import gym
import numpy as np
import random
import signal
import collections
import time

import keras

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


def main(max_episodes=800, print_log_episodes=20):
    env = gym.make('CartPole-v0')

    keyboard_input = KeyboardCtrlC()
    transitions = collections.deque(maxlen=10000)
    episodes = collections.deque(maxlen=100)
    start_time = time.time()

    # Get model definition and define network related items
    model = get_model()

    for episode in range(max_episodes):
        observation = env.reset()

        for iteration in range(201): # iteration stops at 199 because done == True
            old_observation = observation

            epsilon = max(0.025, 0.7 * pow(0.99, episode))

            if np.random.random() < epsilon:
                action = np.random.choice(range(2))
            else:
                # Evaluate the right action, either 0 or 1
                action = np.argmax(model.predict(np.expand_dims(old_observation, 0)))

            observation, reward, done, info = env.step(action)

            if done and iteration < 199:
                reward = -50 # we failed, punish when game ends too early

            transitions.append([old_observation, action, reward, observation])

            # Training of the network, first calculate future reward and then train the network
            minibatch_size = 64
            if len(transitions) >= minibatch_size*4 and iteration % 10 == 0:
                sample_transitions = np.array(random.sample(transitions, minibatch_size))

                train_x = np.array([np.array(x) for x in sample_transitions[:,0]])
                train_y_target = model.predict(train_x)

                next_state_x = np.array([np.array(x) for x in sample_transitions[:,3]])
                next_state_value_y = model.predict(next_state_x)

                for idx, arr in enumerate(sample_transitions):
                    state, action, reward, next_state = arr

                    if reward > 0: # game did not end, add discounted future reward
                        gamma = 0.99
                        reward += gamma * next_state_value_y[idx].max()

                    train_y_target[idx, action] = reward

                # note that keras model hides the original train_y, so we are doing the predicting again
                model.fit(train_x, train_y_target, batch_size=minibatch_size, epochs=1, verbose=0, shuffle=False)

            if keyboard_input.key_pressed:
                print("Started python console. Quit with 'ctrl-d' or continue with 'c'")
                import ipdb; ipdb.set_trace()

            if done:
                break

        episodes.append(iteration)
        if episode % print_log_episodes == 0:
            print('Episode {}, last-100-avg-reward: {}, epsilon: {:.2f}, seconds: {:.2f}'.format(episode, sum(episodes)//len(episodes), epsilon, time.time()-start_time))
        if sum(episodes)/len(episodes) > 195:
            print('SOLVED AT EPISODE {}, time: {:.2f}s'.format(episode, time.time()-start_time))
            return episode


if __name__ == "__main__":
    main()
