
import gym
import numpy as np
import random
import signal
import collections
import time

import tensorflow as tf

class KeyboardCtrlC:
    def __init__(self):
        self.key_pressed = False
        signal.signal(signal.SIGINT, self.key_pressed_m)
        signal.signal(signal.SIGTERM, self.key_pressed_m)

    def key_pressed_m(self, signum, frame):
        self.key_pressed = True


def get_model(x_ph):
    x = tf.layers.dense(inputs=x_ph, units=64, activation=tf.nn.relu, name='dense1')
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name='dense2')
    x = tf.layers.dense(inputs=x, units=2)
    y_results = tf.identity(x, name='y_results') # renaming the last layer
    return y_results


def main(max_episodes=800, print_log_episodes=20):
    env = gym.make('CartPole-v0')

    keyboard_input = KeyboardCtrlC()
    transitions = collections.deque(maxlen=1000)
    episodes = collections.deque(maxlen=100)
    start_time = time.time()

    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, [None, 4], name='x_ph')
    y_results = get_model(x_ph)
    y_targets_ph = tf.placeholder(tf.float32, [None, 2], name='y_targets_ph')
    loss_op = tf.losses.mean_squared_error(labels=y_targets_ph, predictions=y_results)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for episode in range(max_episodes):
        observation = env.reset()

        for iteration in range(201): # iteration stops at 199 because done == True
            old_observation = observation

            epsilon = max(0.025, 0.7 * pow(0.99, episode))

            if np.random.random() < epsilon:
                action = np.random.choice(range(2))
            else:
                # action is integer, either 0 or 1
                action = np.argmax(sess.run([y_results], feed_dict={x_ph: np.expand_dims(old_observation, 0)})[0])

            observation, reward, done, info = env.step(action)

            if done and iteration < 199:
                reward = -50 # we failed, punish when game ends too early

            transitions.append([old_observation, action, reward, observation])

            minibatch_size = 64
            if len(transitions) >= minibatch_size*4 and iteration % 10 == 0:
                sample_transitions = np.array(random.sample(transitions, minibatch_size))

                train_x = np.array([np.array(x) for x in sample_transitions[:,0]])
                train_y = sess.run([y_results], feed_dict={x_ph: train_x})[0]

                next_state_x = np.array([np.array(x) for x in sample_transitions[:,3]])
                next_state_value_y = sess.run([y_results], feed_dict={x_ph: next_state_x})[0]

                for idx, arr in enumerate(sample_transitions):
                    state, action, reward, next_state = arr

                    if reward > 0: # game did not end, add discounted future reward
                        gamma = 0.99
                        reward += gamma * next_state_value_y[idx].max()

                    train_y[idx, action] = reward

                _, loss = sess.run([train_op, loss_op], feed_dict={x_ph: train_x, y_targets_ph: train_y})

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
