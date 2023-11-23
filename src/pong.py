# http://karpathy.github.io/2016/05/31/rl/
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import numpy as np
import gym
import pickle

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 0.001
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80 x 80 grid

if resume:
  model = pickle.load(open('model/save.p', 'rb'))
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
  discounted_r = np.zeros_like(r).astype('float64')
  summed_up_r = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0:
      summed_up_r = 0 # One player has won, so we need to reset
    summed_up_r = r[t] + gamma * summed_up_r
    discounted_r[t] = summed_up_r
  return discounted_r

def policy_backward(epx, eph, edplogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  # dC/dW2 = dC/dZ2 * dZ2/dW2 = dC/dZ2 * A1
  # = dC/dA2 * dA2/dZ2 * A1
  # with A2 = sigmoid(Z2) = aprob and C = log(A2)
  # = 1/aprob * aprob * (1-aprob) * A1
  # = edplogp * A1
  # with A1 = eph
  dW2 = np.dot(eph.T, edplogp).ravel()
  # dC/dW1 = dC/dZ1 * dZ1/dW1 = dC/dZ1 * A0
  # = dC/dA1 * dA1/dZ1 * A0
  # = dC/dA1 * dReLu(Z1) * A0
  # = dC/dZ2 * dZ2/dA1 * dReLu(Z1) * A0
  # = dC/dZ2 * W2 * dReLu(Z1) * A0
  # with dC/dZ2 = (1-aprob) = edplogp and dReLu(Z1) = 1 for Z1>0 and 0 for Z1<0
  # = edplogp * W2 * dReLu* A0
  dh = np.outer(edplogp, model['W2'])
  dh[eph<=0] = 0 # Derivative of ReLu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make('ALE/Pong-v5', render_mode='human')

observation = env.reset()

observation = observation[0]
prev_x = None # used for computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render:
    env.render()

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

    # perform rmsprop parameter update every batch_size episodes
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

    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5) # 1e-5 is added to prevent diff by 0
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

