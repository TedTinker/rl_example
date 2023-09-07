#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import gym as gymnasium
import matplotlib.pyplot as plt

from utils import ReplayBuffer, plot_durations
from model import DQN

# Constants and Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.005
GAMMA = 0.99
RANDOM_ACTION_THRESHOLD_INIT = 1
RANDOM_ACTION_DECAY = 0.9999
RANDOM_ACTION_THRESHOLD = RANDOM_ACTION_THRESHOLD_INIT
EPISODES = 10000

# Model, buffer, and optimizer setup
buffer = ReplayBuffer()
policy_dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
optimizer = optim.Adam(policy_dqn.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

# Environment setup
env = gymnasium.make("CartPole-v1")
episode_durations = []



def train():
    """
    Train the model using experience from the replay buffer.
    """
    state, action, next_state, reward, done = buffer.sample()
    
    # Predicted values: Q^\pi (s_t, a_t)
    state_action_values = policy_dqn(state).gather(1, action)
    
    # Target values: r + \gamma Q^\pi(s_t+1, \pi(s_t+1))
    with torch.no_grad():
        next_state_values = target_dqn(next_state).max(1)[0]
        next_state_values *= ~done
        expected_state_action_values = (next_state_values * GAMMA) + reward
    
    # Compute loss and optimize
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 100)
    optimizer.step()
    
    # Soft update target network: \bar{\theta} <- \tau\theta + (1 - \tau) \bar{\theta}
    target_dqn_state_dict = target_dqn.state_dict()
    policy_dqn_state_dict = policy_dqn.state_dict()
    for key in policy_dqn_state_dict:
        target_dqn_state_dict[key] = policy_dqn_state_dict[key] * TAU + target_dqn_state_dict[key] * (1 - TAU)
    target_dqn.load_state_dict(target_dqn_state_dict)
    
    

def select_action(state):
    """
    Select an action, either randomly or based on the current state.
    """
    global RANDOM_ACTION_THRESHOLD

    if random.random() > RANDOM_ACTION_THRESHOLD:
        with torch.no_grad():
            action = policy_dqn(state)
            print(action)
            action = action.max(1)[1].view(1, 1)
    else:   action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    RANDOM_ACTION_THRESHOLD *= RANDOM_ACTION_DECAY

    return action



def run_episode(render = False, episode_num = 0):
    """
    Execute one episode of the environment.
    """
    state = env.reset()
    state = torch.tensor(state, device=device).unsqueeze(0)
    t = 0
    done = False

    while not done:
        if render: 
            frame = env.render(mode="rgb_array")
            plt.axis('off')
            plt.imshow(frame)
            plt.savefig(f'plots/{episode_num}/frame_{t}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        t += 1
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, device=device).unsqueeze(0)
        reward = torch.tensor([-1 if done else 1], device=device)
        buffer.push(state, action, next_state, reward, torch.tensor(done).unsqueeze(0))
        state = next_state
        train()
        
        if done:
            episode_durations.append(t + 1)
            
            

if __name__ == '__main__':
    for episode in range(EPISODES):
        print(episode, end = '... ')
        if episode % 25 == 0:
            try: os.mkdir('plots/{}'.format(episode))
            except: pass
            run_episode(render = True, episode_num = episode)
            plot_durations(episode_durations, episode)
        else:
            run_episode()
# %%
