# gym stuff
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# import helpers
import numpy as np
import os

# import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

class ReblathEnv(Env):
    def __init__(self):
        self.action_space = MultiDiscrete([4,5])
        self.observation_space = Dict({'Reblaths':MultiDiscrete([41,41,41,41]),'Failstacks':MultiDiscrete([401,401,401,401,401])})
        self.state = {'Reblaths':np.array([40,0,0,0]),'Failstacks':np.array([20,20,20,20,20])}
        self.total_taps = 8192
        self.stacks_hit = 0
        self.starti = 20
        
    def step(self, action):
        reward = 0 # initializes the reward
        # perform the action
        if self.state['Reblaths'][action[0]]>0: # if the number of reblaths in the selected reblath type is greater than 0
            # self.state['Failstacks'][action[1]] # current selected failstack amount
            # self.state['Reblaths'][action[0]] # current selected reblath amount
            # action[0] # current selected reblath tier
            match action[0]: # observes what tier of reblath you're enhancing, determines the base enhancement chance
                case 0: basechance = .076923
                case 1: basechance = .062500
                case 2: basechance = .020000
                case 3: basechance = .003000
            enhancementchance = basechance + .1*basechance*self.state['Failstacks'][action[1]] #determines the enhancement chance
            if np.random.random(1)[0] < enhancementchance: # if the enhancement succeeds
                if action[0]==3: # and tet->pen reblath succeeded
                    self.state['Reblaths'][action[0]] -= 1 # decrease the current amount of reblath in that tier by 1
                    self.state['Reblaths'][0] += 1 #obtain another pri reblath
                    #reward += 70-self.state['Failstacks'][action[1]]
                    reward += enhancementchance*(70-self.state['Failstacks'][action[1]])+(1-enhancementchance)*(3+action[0])
                    self.state['Failstacks'][action[1]] = 70 # set the failstack to 70 (devour)
                else: # and it was any other tier of reblath that succeeded
                    self.state['Reblaths'][action[0]] -= 1 # decrease the current amount of reblath in that tier by 1
                    self.state['Reblaths'][action[0]+1] += 1 # increase the current amount of reblath in next tier by 1
                    #reward += 20-self.state['Failstacks'][action[1]]
                    reward += enhancementchance*(self.starti-self.state['Failstacks'][action[1]])+(1-enhancementchance)*(3+action[0])
                    self.state['Failstacks'][action[1]] = self.starti # set the failstack to the starting stack
            else: # if the enhancement fails
                if action[0]==0: #and it was pri->duo reblath that failed
                    reward += 3 + action[0]
                    self.state['Failstacks'][action[1]] += 3+action[0] # increase failstack by 3+index (3,4,5,6)
                else: # and it was any other tier of reblath that failed
                    self.state['Reblaths'][action[0]] -= 1 # decrease the current amount of reblath in that tier by 1
                    self.state['Reblaths'][action[0]-1] += 1 # increase the current amount of reblath in previous tier by 1
                    reward += 3 + action[0]
                    self.state['Failstacks'][action[1]] += 3+action[0] # increase failstack by 3+index (3,4,5,6)
        else: #if the number of reblaths in the selected reblath type is not greater than 0
            reward -= 90 # cant enhance something that doesnt exist! bad!
            #self.total_taps +=1

        if self.state['Failstacks'][action[1]] >= 110: # if the bot actually made a 110 stack with this action 
            reward += 110 # gives a good reward payout
            self.state['Failstacks'][action[1]] = 20 # sets the failstack value to 20
            self.stacks_hit += 1
        
        self.total_taps -= 1
        if self.total_taps <=0:
            done = True
        else:
            done = False
        self.state['Failstacks'].sort()
        info = {"Stacks Hit":[self.stacks_hit],'Failstack List':[self.state['Failstacks'][0],self.state['Failstacks'][1],self.state['Failstacks'][2],self.state['Failstacks'][3],self.state['Failstacks'][4]],"Reb List":[self.state['Reblaths'][0],self.state['Reblaths'][1],self.state['Reblaths'][2],self.state['Reblaths'][3]]}
        
        #reward = reward/116 # this is the reward normalization - impliment after experimenting with backpropagation
        
        return self.state, reward, done, info


    def reset(self):
        self.state = {'Reblaths':np.array([40,0,0,0]),'Failstacks':np.array([20,20,20,20,20])}
        self.total_taps = 8192
        self.stacks_hit = 0
        self.state['Failstacks'].sort()
        return self.state
    def testcase(self):
        self.state = {'Reblaths':np.array([0,0,0,40]),'Failstacks':np.array([20,20,20,80,105])}
        self.total_taps = 8192
        self.stacks_hit = 0
        self.state['Failstacks'].sort()
        return self.state

env = ReblathEnv()

check_env(env)

#testcase the model
episodes = 20
for episode in range(1,episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}  {}'.format(episode,score,info))

log_path = os.path.join('training','logs')
model = PPO('MultiInputPolicy',env, verbose=0,tensorboard_log=log_path,learning_rate=.0005)

#train model
model.learn(total_timesteps=10000)

#save the model
reblath_path = os.path.join('training','models','reblath_PPO_average_10kstep')
model.save(reblath_path)

#test the model
episodes = 10
for episode in range(1,episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        action = model.predict(obs)[0] # now using model here
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{} {}'.format(episode,score,info))
env.reset()

#evaluate model
evaluate_policy(model,env,n_eval_episodes=5)

