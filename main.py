import gymnasium as gym
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.optim as optim
from deepqnetwork import MLP
import math

def evaluate(model, env, num_episodes=10, epsilon=0.05):
    total_rewards = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(torch.tensor(observation).to(device)).argmax().item()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return sum(total_rewards) / num_episodes


    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()
# print(observation)
# print(env.action_space)
# epsilon = 0.7
N = 5000
gamma = 0.99
memory = []
rewards = []
actions_batch = []
#Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
#memory = torch.tensor([])
action_value = MLP(input_dim=8, output_dim=4).to(device)
action_value.requires_grad = True
target = MLP(input_dim=8, output_dim=4).to(device)
target.load_state_dict(action_value.state_dict())
target.requires_grad = False
n_episodes = 600
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(action_value.parameters(), lr = 2e-3)
C = 0
total_loss = 0
total_rewards = 0
steps_done = 0
tau = 0.005  # Soft update parameter

for n in range(n_episodes):
    print(f'New Episode : {n}')
    observation = torch.tensor(observation).to(device)
    sequence = [observation]
    count = 0
    total_loss = 0
    total_rewards = 0
    while True:
        rand = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        if rand <= eps_threshold:
            action = env.action_space.sample()  # agent policy that uses the observation and info
        else:
            with torch.no_grad():
                action = action_value(observation)
                # print(action.grad_fn)
                action = torch.argmax(action)
                # print(action.grad_fn)
        # action = action.cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action.item())
        total_rewards += reward
        # print(next_observation)
        # print(terminated)
        if terminated or truncated:
            next_observation = None
        else:
            next_observation = torch.tensor(next_observation).to(device)
            
        # reward = torch.tensor(reward)
        # action = torch.tensor(action)
        sequence.append(next_observation)
        observation = observation.tolist()
        memory.append((observation, action, reward, next_observation))
        # rewards.append(reward)
        #actions.append(action)
        # print(mem)
        # if len(mem) < 2:
        #     continue
        # memory = torch.tensor(mem)
        observation = next_observation
        num_indices = 128
        if len(memory) < num_indices:
            if terminated or truncated:
                observation, info = env.reset()
                break
            continue
        episode = random.sample(memory, num_indices)
        
        observations, actions, rewards, next_observations = zip(*episode)
        non_final_state = torch.tensor(tuple(map(lambda s: s is not None, next_observations)), device = device, dtype = torch.bool)
        non_final_observations = torch.tensor([list(s) for s in next_observations if s is not None], device=device)
        #print(non_final_observations.shape)
        rewards = torch.tensor(list(rewards)).to(device)
        observations = torch.tensor(list(observations)).to(device)
        actions = torch.tensor(list(actions)).to(device)
        #print(actions)
        #print(observations)
        #observations = torch.cat(observations)
        #print(observations)
        # print(observations.shape)
        # minibatch = memory[random_indices] # can also use torch.utils.Subset
        # print(minibatch[:][2])
        # print(target(observation))
        # minibatch = list(operator.itemgetter(*indices)(memory))
        # reward_batch = list(operator.itemgetter(*indices)(reward))
        grad = torch.zeros((128), device=device)
        # print(non_final_state.shape)
        # print(grad.shape)
        # print(torch.argmax(target(observations), dim=-1).float())
        # print(grad[non_final_state].shape)
        with torch.no_grad():
            grad[non_final_state] = torch.argmax(target(non_final_observations), dim=-1).float()
        grad = rewards + gamma * grad
        # print(actions.shape)
        # print(action_value(observations).shape)
        state_action = action_value(observations).gather(1, actions.unsqueeze(1))
        # print(state_action)
        # state_action = torch.argmax(state_action, dim=-1)
        # print(state_action)
        # print(state_action.shape)
        # print(grad.shape)
        loss = criterion(state_action.squeeze(1), grad)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_value_(action_value.parameters(), 100)
        optimizer.step()

        # if C == 500:
        #     target.load_state_dict(action_value.state_dict())
        #     C = 0
        # else:
        #     C += 1
        # count += 1

        # Soft update of the target network
        for target_param, param in zip(target.parameters(), action_value.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        count += 1
        

        if terminated or truncated:
            observation, info = env.reset()
            break
    if count:
        print(f'average loss : {total_loss/count}')
        print(f'total reward : {total_rewards}')
    if n % 10 == 0:
        with torch.no_grad():
            avg_reward = evaluate(action_value, env)
            print(f'Evaluation Average Reward at Episode {n}: {avg_reward}')

torch.save(action_value.state_dict(), 'policy_net')
env.close()