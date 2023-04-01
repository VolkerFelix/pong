# http://karpathy.github.io/2016/05/31/rl/
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import numpy as np
import gym
import pickle
import pprint

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

# Creates a dict with key, value pairs  
grad_buffer = { key : np.zeros_like(value) for key,value in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { key : np.zeros_like(value) for key,value in model.items() } # rmsprop memory

def sigmoid(f_x): 
  return 1.0 / (1.0 + np.exp(-f_x)) # sigmoid "squashing" function to interval [0,1]

def prepro(f_field_input):
  """ pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  cropped_field = f_field_input[35:195] # crop
  cropped_field_down = cropped_field[::2,::2,0] # downsample by factor of 2
  # Keep in mind that the input will be a numpy array,
  # meaning that the == allows to do an element-wise comparison with a scalar
  cropped_field_down[cropped_field_down == 144] = 0 # erase background (background type 1)
  cropped_field_down[cropped_field_down == 109] = 0 # erase background (background type 2)
  cropped_field_down[cropped_field_down != 0] = 1 # everything else (paddles, ball) just set to 1
  return cropped_field_down.astype(float).ravel()

squares = np.array([[[x for x in range(3)] for y in range(210)] for z in range(160)])

print('gym:', gym.__version__)

env = gym.make("Pong-v0")
observation = env.reset()
pprint.pprint(observation.shape)
