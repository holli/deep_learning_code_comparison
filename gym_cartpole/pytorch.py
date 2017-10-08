# Pytorch implementation, will return about the following:


import gym
import numpy as np
import random
import signal
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

class KeyboardCtrlC:
    def __init__(self):
        self.key_pressed = False
        signal.signal(signal.SIGINT, self.key_pressed_m)
        signal.signal(signal.SIGTERM, self.key_pressed_m)

    def key_pressed_m(self, signum, frame):
        self.key_pressed = True


def get_model(obs=4):

    model = nn.Sequential(
                nn.Linear(obs, 16),
                torch.nn.ReLU(),
                nn.Linear(16, 16),
                torch.nn.ReLU(),
                nn.Linear(16, 2)
    )
    return model


def main(max_episodes, print_log_episodes=20):
    env = gym.make('CartPole-v0')

    keyboard_input = KeyboardCtrlC()
    transitions = collections.deque(maxlen=1000)
    episodes = collections.deque(maxlen=100)
    start_time = time.time()

    model = get_model().double()#.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # loss_op = torch.nn.MSELoss(size_average=True)
    loss_op = torch.nn.MSELoss()

    for episode in range(max_episodes):
        observation = env.reset()

        for iteration in range(300):
            old_observation = observation

            epsilon = max(0.1, 0.7 * pow(0.985, episode))

            if np.random.random() < epsilon:
                action = np.random.choice(range(2))
            else:
                # action is integer, either 0 or 1
                # x = old_observation.astype(np.float32)
                x = old_observation
                x = Variable(torch.from_numpy(np.expand_dims(x, 0)))#.cuda()
                action = model(x)[0].max(0)[1].data[0] # same as np.argmax(model(x)[0].data.numpy()) but works both for cpu and gpu

            observation, reward, done, info = env.step(action)

            if done and iteration < 199:
                reward = -10 # we failed, punish when game ends too early

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
            train_x = Variable(torch.from_numpy(train_x))#.cuda()
            train_y = model(train_x)

            next_state_x = np.array([np.array(x) for x in sample_transitions[:,3]])
            next_state_x = Variable(torch.from_numpy(next_state_x))#.cuda()
            next_state_value_y = model(next_state_x)

            for idx, arr in enumerate(sample_transitions):
                state, action, reward, next_state = arr

                if reward > 0: # normal round, add discounted future reward
                    gamma = 0.99
                    reward += gamma * next_state_value_y[idx].max().data[0]

                train_y[idx, action] = reward

            predict = model(train_x)
            loss = loss_op(predict, train_y.detach())
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    results = []
    result = main(1000)
    print("Pytorch done")

