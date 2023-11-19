#criando classe Q learning para usar no ambinete flappy bird
#reference: https://aspram.medium.com/learning-flappy-bird-agents-with-reinforcement-learning-d07f31609333


#import flappy_bird_gymnasium
import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn
from scipy import array, newaxis
from IPython.display import clear_output
import os, sys
import gymnasium as gym
import time
from tqdm import tqdm
import pickle
from collections import defaultdict
from mpl_toolkits.mplot3d import axes3d


#env = gymnasium.make("FlappyBird-v0", render_mode="human")


env = gym.make('TextFlappyBird-v0',  height = 10, width = 18, pipe_gap = 4)
 
class Agent:
    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'act'
    method
    """
    
    ## take random action if random float is less than epsilon
    ## otherwise select action with highest Q-score for the given state
        
    def act(self, state): #epsilon-greedy policy
        
        
        if np.random.uniform(0, 1) < self.eps:
            action = env.action_space.sample()
            return action
        else:
            action = np.argmax(q[state])
            return action


class Q_Agent(Agent):
    """
    The Q-Learning Agent Class
    
    """        
    
    def __init__(self, eps, step_size, discount):
        self.eps = eps
        self.step_size = step_size
        self.discount = discount
        
    def update(self):
        old_value = q[state][action]
        next_max = np.max(q[next_state])
        
        new_value = (1 - self.step_size) * old_value + self.step_size * (reward +  self.discount * next_max)
        q[state][action] = new_value

class SarsaAgent(Agent):
    """
    The SARSA Agent Class
    
    """    
    
    def __init__(self, eps, step_size, discount):
        self.eps = eps
        self.step_size = step_size
        self.discount = discount
        
    def update(self):
                
        q[state][action]+= self.step_size * \
        (reward + self.discount * (q[next_state][next_action]) - q[state][action])

class RandomAgent():
    """
    The Random Agent Class
    
    """
    def __init__(self, eps, step_size, discount):
        self.eps = eps
        self.eps=1
        self.step_size = step_size
        self.discount = discount
        
        
    def act(self, state): 
        
        action = env.action_space.sample()
        return action
    def update(self):
        pass

agents = {
    "Q-learning": Q_Agent,
    "Sarsa": SarsaAgent,
    "RandomAgent": RandomAgent}
all_the_q_tables = {}
all_reward_sums = {} # Contains sum of rewards during episode
for algorithm in ["Q-learning", "Sarsa", "RandomAgent"]:
    
    q = defaultdict(lambda: np.zeros(2))  # The dict of action-value estimates.
    
    all_reward_sums[algorithm] = []
    current_agent = agents[algorithm](eps=0.2, step_size = 0.7, discount=0.95)
    #total_epochs = 0

    for i in range(10000):
        
        if i % 100 == 0:
            print("step: ", i)
            
        state = env.reset()[0]
        #print(state)
        done = False
        total_reward = 0
        while not done:

            #os.system("cls")
            #sys.stdout.write(env.render())
            #time.sleep(0.2)

                    
            action = current_agent.act(state)
# Apply action and return new observation of the environment
            next_state, reward, done, truncate, info = env.step(action)

            
            #For SARSA acquiring the on-policy next action
            next_action = current_agent.act(next_state)
            
            """print("next_state: ",next_state)
            print("reward ",reward)
            print("done ",done)
            print("truncate ",truncate)
            print("info ",info)"""

            if done == True:
                reward = -1

                # Update total reward 
            total_reward += reward
#Update q values table
            current_agent.update()
                
            state = next_state
            if done: 
                    break
            all_reward_sums[algorithm].append(total_reward)
            env.close()
            all_the_q_tables[algorithm] = q
        


for algorithm in ["Q-learning", "Sarsa", "RandomAgent"]:
    plt.plot(all_reward_sums[algorithm], label=algorithm)
plt.xlabel("Episodes")
plt.ylabel("Sum of\n rewards\n during\n episode",rotation=0, labelpad=40)
plt.xlim(0,10000)
plt.ylim(0,400)
plt.legend()
plt.show()

"""

STEPS = 10000

Qql = defaultdict(lambda: np.zeros(2))
all_reward_sums = {}
current_agent = Q_Agent(eps=0.2, step_size = 0.7, discount=0.95)

# Initialize parameters
#vql = np.zeros(stateEPS)

# Initialize stateate to stateart state (not mandatory, any state is ok)

# Run learning cycle
for t in range(STEPS):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
    
        action = current_agent.act(state)
            
        # Reward
        next_state, rt, terminated, _, info = env.step(action)   
    

    if done == True:

        reward = -1
 
        # Update total reward 
        total_reward += reward

    
    # Update Q
    current_agent.update()

    state = next_state
    
    #Qql[st,at]=next_stage

    if rt == 1.0:
        state = np.random.choice(next_state)
    else:
        state = next_state
        
    vql[t] = la.norm(Qql)

                      
plt.figure()
#plt.plot(vmbl, '-b', label='Model-based')
plt.plot(vql, '--r', label='Q-learning')
plt.xlabel('N. iterations')
plt.ylabel(r'$||Q^*-Q_t||$')
plt.legend(loc='best')
plt.show()"""


