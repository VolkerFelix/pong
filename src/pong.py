# http://karpathy.github.io/2016/05/31/rl/
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import numpy as np
import gym
import pickle

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

def sigmoid(f_x: float) -> float:
  """Sigmoid function"""
  return 1.0 / (1.0 + np.exp(-f_x)) # squash values to [0, 1]

def preprocess_frame(f_frame_input: np.array) -> np.array:
  """Pre-process a 210 x 160 x 3 rgb frame into 6400 (80 x 80) 1D float vector"""
  cropped_field = f_frame_input[CROP_LEFT : (210 - CROP_RIGHT)] # crop width
  cropped_field_down = cropped_field[::2, ::2, 0] # downsample by factor of 2 and take only the "R" value
  cropped_field_down[cropped_field_down == 144] = 0 # erase background (background type 1)
  cropped_field_down[cropped_field_down == 109] = 0 # erase background (background type 2)
  cropped_field_down[cropped_field_down != 0] = 1 # everything else (paddles, ball) just set to 1
  return cropped_field_down.astype(float).ravel()

def policy_forward(f_x: np.array) -> (float, np.array):
  """Forward path through the neural net, which serves as the policy"""
  h = np.dot(model['W1'], f_x)
  h[h<0] = 0 # ReLu activation
  logp = np.dot(model['W2'], h) # Log probability
  p = sigmoid(logp) # Probability between 0 and 1
  return p, h # return probability of taking action 2 (UP), and hidden state

if RESUME:
  model = pickle.load(open('model/save.p', 'rb'))
else:
  model = initialize_model(HIDDEN_NEURONS, CROPPED_FRAME_DIM)

# Creates a dict with key, value pairs  
grad_buffer = { key : np.zeros_like(value) for key, value in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { key : np.zeros_like(value) for key,value in model.items() } # rmsprop memory

def discounted_rewards(f_rewards: np.array) -> np.array:
  """Calculates the discounted rewards of an episode backwards in time"""
  # G_t = R_t + GAMMA * R_t+1 + GAMMA^2 * R_t+2 + GAMMA^3 * R_t+3 + ...
  # G_t = R_t + GAMMA ( R_t+1 + GAMMA * R_t+2 + GAMMA^2 * R_t+3 + ... )
  # In order to calculate this iteratively, we need to run it "backwards" in time
  # G_new = R + GAMMA * G_prev
  discounted_rewards = np.zeros_like(f_rewards).astype('float64')
  summed_up_rewards = 0
  for t in reversed(range(0, f_rewards.size)):
    if f_rewards[t] != 0:
      # if reward = +1 -> You won
      # if reward = -1 -> OpenAI bot won
      # --> need to reset
      summed_up_rewards = 0
    summed_up_rewards = f_rewards[t] + GAMMA * summed_up_rewards
    discounted_rewards[t] = summed_up_rewards
  return discounted_rewards

def policy_backward(f_epx, f_eph, f_edplogp):
  """Backward pass. (eph is array of intermediate hidden states)"""
  # dC/dW2 = dC/dZ2 * dZ2/dW2 = dC/dZ2 * A1
  # = dC/dA2 * dA2/dZ2 * A1
  # with A2 = sigmoid(Z2) = aprob and C = log(A2)
  # = 1/aprob * aprob * (1-aprob) * A1
  # = edplogp * A1
  # with A1 = eph
  dW2 = np.dot(f_eph.T, f_edplogp).ravel()
  # dC/dW1 = dC/dZ1 * dZ1/dW1 = dC/dZ1 * A0
  # = dC/dA1 * dA1/dZ1 * A0
  # = dC/dA1 * dReLu(Z1) * A0
  # = dC/dZ2 * dZ2/dA1 * dReLu(Z1) * A0
  # = dC/dZ2 * W2 * dReLu(Z1) * A0
  # with dC/dZ2 = (1-aprob) = edplogp and dReLu(Z1) = 1 for Z1>0 and 0 for Z1<0
  # = edplogp * W2 * dReLu* A0
  dh = np.outer(f_edplogp, model['W2'])
  dh[f_eph<=0] = 0 # Derivative of ReLu
  dW1 = np.dot(dh.T, f_epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make('ALE/Pong-v5', render_mode='human')

observation, info = env.reset()

#observation = observation[0]
prev_x = None # used for computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

if __name__ == "__main__":

  check_dimensions_match()

  while True:
    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess_frame(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(CROPPED_FRAME_DIM)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability  
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state

    # Background:
    # Classic policy gradient algo = REINFORCE
    # gradient(Expectaion[r(tau)] = Expectation[(SUM(G_t * gradient * log * PI_teta(a_t|s_t)))]
    # With PI_teta: Policy PI with its params teta
    # PI_teta is the output of the 2-layer policy network: aprob
    # Hint: In order to get rid of the expectation, we sample over a large number of trajectories
    # and average them out. This technique is called Markov Chain Monte-Carlo.
    #
    # Calculate the gradient * log * PI_teta:
    # aprob = sigmoid(in)
    # if action == 2:
    # logprob = log(aprob)
    # dlogprob/daprob = dlog(aprob)/daprob * daprob/din
    # = 1/aprob * d_aprob/d_in = 1/aprob * sigmoid(in)(1-sigmoid(in))
    # = 1 - aprob
    # if action == 3:
    # logprob = log(1-aprob)
    # dlogprob/daprob = dlog(1-aprob)/daprob * daprob/din
    # = -1/(1-aprob) * sigmoid(in)(1-sigmoid(in))
    # = -aprob/(1-aprob)*(1-aprob) = -aprob = 0 - aprob
    y = 1 if action == 2 else 0 # fake label

    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

    # step the environment and get new measurements
    observation, reward, done, truncated, info = env.step(action)

    reward_sum += reward

    drs.append(reward)

    if done:
      # an episode has finished (= one player reached a score of 21)
      episode_number += 1

      # stack together all inputs, hidden states, action gradients, and rewards for this episode
      epx = np.vstack(xs)
      eph = np.vstack(hs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      xs,hs,dlogps,drs = [],[],[],[] # reset array memory

      # compute the discounted reward
      discounted_epr = discounted_rewards(epr)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
      grad = policy_backward(epx, eph, epdlogp)

      for k in model:
        grad_buffer[k] += grad[k] # accumulate gradient over batch

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
          grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

      # book-keeping
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print('Resetting env. Episode reward total was {}. Running mean: {}'.format(reward_sum, running_reward))
      if episode_number % 100  == 0:
        pickle.dump(model, open('save.p', 'wb'))

      reward_sum = 0
      observation = env.reset() # reset env
      # For gym ~ 0.26
      observation = observation[0]
      prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when the game ends.
      print('Episode number {}: Game finished with reward: {}'.format(episode_number, reward) + ('' if reward == -1 else '!!!!!!!'))

