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

def policy_forward(x):
  """forward path through the neural net which serves as the policy"""
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLu activation
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp) # Prob between 0 and 1
  return p, h # return probability of taking action 2, and hidden state

def discounted_rewards(r):
  """calculates the discounted rewards of an episode backwards in time"""
  # G_t+1 = R + gamma * G_t
  # the reason to calculate it backwards in time is that it enables us to use an iterative algo
  discounted_r = np.zeros_like(r)
  summed_up_r = 0
  for t in reversed(range(0, r.size)):
    if t != 0:
      summed_up_r = 0 # One player has won, so we need to reset
    else:
      summed_up_r = r[t] + gamma * summed_up_r
    discounted_r[t] = summed_up_r
  return discounted_r

env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability  
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # fake label
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)
  reward_sum += reward
  drs.append(reward_sum)

  if done:
    # an episode has finished (= one player reached a score of 21)
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    eph1 = np.vstack(h1s)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,h1s,dlogps,drs = [],[],[],[],[] # reset array memory

    # compute the discounted reward
    discounted_epr = discounted_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)