import numpy as np

def sigmoid(f_x: float) -> float:
  """Sigmoid function"""
  return 1.0 / (1.0 + np.exp(-f_x)) # squash values to [0, 1]

def policy_forward(f_model: dict, f_x: np.array) -> (float, np.array):
  """Forward path through the neural net, which serves as the policy"""
  h = np.dot(f_model['W1'], f_x)
  h[h<0] = 0 # ReLu activation
  logp = np.dot(f_model['W2'], h) # Log probability
  p = sigmoid(logp) # Probability between 0 and 1
  return p, h # return probability of taking action 2 (UP), and hidden state

def policy_backward(f_model: dict, f_epx: np.array, f_eph: np.array, f_epdlogp: np.array) -> dict:
  """
  Backward pass
  
  :param dict f_model: Neural net model
  :param np.array f_epx: Inputs to NN (= X in maths)
  :param np.array f_eph: Intermediate hidden states (= A1 in maths)
  :param np.array f_epdlogp: Gradient of logp
  """
  dW2 = np.dot(f_eph.T, f_epdlogp).ravel()
  dh = np.outer(f_epdlogp, f_model['W2'])
  dh[f_eph<=0] = 0 # Derivative of ReLu
  dW1 = np.dot(dh.T, f_epx)
  return {'W1':dW1, 'W2':dW2}

def discounted_rewards(f_rewards: np.array, f_gamma: float) -> np.array:
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
    summed_up_rewards = f_rewards[t] + f_gamma * summed_up_rewards
    discounted_rewards[t] = summed_up_rewards
  return discounted_rewards