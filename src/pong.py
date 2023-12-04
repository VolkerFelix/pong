# http://karpathy.github.io/2016/05/31/rl/
"""Trains an agent with (stochastic) Policy Gradients on Pong"""

import numpy as np
import gym
import pickle
import helpers

# Hyperparameters
HIDDEN_NEURONS = 200 # Number of hidden layer neurons
BATCH_SIZE = 10 # Param update after how many episodes?
LEARNING_RATE = 0.001
GAMMA = 0.99 # Discount factor for reward
DECAY_RATE = 0.99 # Decay factor for RMSProp leaky sum of grad^2

# Configuration
RESUME = True # Resume from previous checkpoint?
CROPPED_FRAME_DIM = 80 * 80
CROP_LEFT = 35 # Pixels to be cropped counting from the left
CROP_RIGHT = 15 # Pixels to be cropped counting from the right

def check_dimensions_match():
  assert (210 - CROP_LEFT - CROP_RIGHT) == 160, "The cropped frame is not 160px wide."

def initialize_model(f_nr_hidden_neurons: int, f_nr_inputs: int) -> dict:
  """Init the model using "Xavier" initialization"""
  model = {}
  model['W1'] = np.random.randn(f_nr_hidden_neurons, f_nr_inputs) / np.sqrt(f_nr_inputs)
  model['W2'] = np.random.randn(f_nr_hidden_neurons) / np.sqrt(f_nr_hidden_neurons)
  return model

def preprocess_frame(f_frame_input: np.array) -> np.array:
  """Pre-process a 210 x 160 x 3 rgb frame into 6400 (80 x 80) 1D float vector"""
  cropped_field = f_frame_input[CROP_LEFT : (210 - CROP_RIGHT)] # crop width
  cropped_field_down = cropped_field[::2, ::2, 0] # downsample by factor of 2 and take only the "R" value
  cropped_field_down[cropped_field_down == 144] = 0 # erase background (background type 1)
  cropped_field_down[cropped_field_down == 109] = 0 # erase background (background type 2)
  cropped_field_down[cropped_field_down != 0] = 1 # everything else (paddles, ball) just set to 1
  return cropped_field_down.astype(float).ravel()

prev_x = None # used for computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

if __name__ == "__main__":

  if RESUME:
    model = pickle.load(open('model/save.p', 'rb'))
  else:
    model = initialize_model(HIDDEN_NEURONS, CROPPED_FRAME_DIM)

  # Creates a dict with key, value pairs  
  grad_buffer = { key : np.zeros_like(value) for key, value in model.items() } # update buffers that add up gradients over a batch
  rmsprop_cache = { key : np.zeros_like(value) for key,value in model.items() } # rmsprop memory

  env = gym.make('ALE/Pong-v5', render_mode='human')
  observation, info = env.reset()

  check_dimensions_match()

  while True:
    # Preprocess the observation, set input to network to be difference image
    cur_x = preprocess_frame(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(CROPPED_FRAME_DIM)
    prev_x = cur_x

    # Forward the policy network and sample an action from the returned probability  
    aprob, h = helpers.policy_forward(model, x)
    # Introduce randomness for exploration of the agent
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # Record various intermediates (needed later for backprop)
    xs.append(x) # Observation
    hs.append(h) # Hidden state

    y = 1 if action == 2 else 0
    dlogps.append(y - aprob) # Grad that encourages the action that was taken to be taken

    # Step the environment and get new measurements
    observation, reward, done, truncated, info = env.step(action)

    reward_sum += reward
    drs.append(reward)

    if done:
      # An episode has finished (= one player reached a score of 21)
      episode_number += 1

      # Stack together all inputs, hidden states, action gradients, and rewards for this episode
      epx = np.vstack(xs)
      eph = np.vstack(hs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      xs,hs,dlogps,drs = [],[],[],[] # Reset array memory

      # Compute the discounted reward
      discounted_epr = helpers.discounted_rewards(epr, GAMMA)
      # Standardize the rewards to be unit normal (helps control the gradient estimator variance)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlogp *= discounted_epr # Modulate the gradient with advantage (PG magic happens right here.)
      grad = helpers.policy_backward(model, epx, eph, epdlogp)

      for k in model:
        grad_buffer[k] += grad[k] # Accumulate gradient over batch

      # perform rmsprop parameter update every BATCH_SIZE episodes
      # Background on gradient descent:
      # Use backprob to calculate the gradients based on the loss function
      # Update the weights of the model using learning rate alpha:
      # W = W - alpha * dW
      # Background on rmsprop:
      # We want the gradients to oscillate less and move faster in the horizonal direction
      # towards the minimum for a 2-D problem. Rmsprop works similarly as gradient descent 
      # with momentum, which uses the exponentially weighted averages of the gradients
      # in order to dampens out the oscillations.
      # https://www.youtube.com/watch?v=_e-LFe_igno
      # Rmsprop formula:
      # S_dw = beta * S_dw + (1-beta) * dw^2
      # W = W - alpha * dw / sqrt(S_dw)

      if episode_number % BATCH_SIZE == 0:
        for k,v in model.items():
          g = grad_buffer[k] # gradient
          rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g**2
          model[k] += LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5) # 1e-5 is added to prevent diff by 0
          grad_buffer[k] = np.zeros_like(v) # Reset batch gradient buffer

      # book-keeping
      running_reward = reward_sum if running_reward is None else running_reward * GAMMA + reward_sum * (1 - GAMMA)
      print('Resetting env. Episode reward total was {}. Running mean: {}'.format(reward_sum, running_reward))
      if episode_number % 100  == 0:
        pickle.dump(model, open('save.p', 'wb'))

      reward_sum = 0
      observation, info = env.reset()
      prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when the game ends.
      print('Episode number {}: Game finished with reward: {}'.format(episode_number, reward) + ('' if reward == -1 else '!!!!!!!'))

