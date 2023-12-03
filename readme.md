# Trains an agent with (stochastic) Policy Gradients on Pong.
Based on this awesome blog post from the great Andrej Kaparthy

https://karpathy.github.io/2016/05/31/rl/

## Architecture
![model_arch](architecture/myfile.dot.png)
Weights  
W1: (200 x 6400)  
W2: (1 x 200)

## Calculations
### Forward path
Z1 = dot_prod(W1, X)  
A1 = ReLU(Z1) *(≡ "h" in code)*  
Z2 = dot_prod(W2, A1) *(≡ "logp" in code)*  
A2 = sigmoid(Z2) *(≡ "p" in code)*

### Backward path
#### Statistics
Since our final output of our forward calculations is a probability of sampling the action of going UP (=1), basically a coin toss, we can make use of the [Bernoulli Distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution):  
p(x, theta) = theta^x (1 - theta)^(1 - x)  
The log-likelihood function is:  
logL(theta) = sum(i=1..n)(xi * log(theta)) + sum(i=1..n)((1 - xi) * log(1 - theta))  
Keep in mind that all our efforts during training focus on optimizing theta (represented by the 2-layer NN), in order to let us win as many games as possible.  
Our loss-function that we want to minimize is logL for n=1. Theta is represented by A2 or "p" in the code.  
Calculate the partial derivatives:  
dlogL / dW2  
= dC / dZ2 * dZ2 / dW2 = dC / dZ2 * A1  
= dC / dA2 * dA2 / dZ2 * A1  


## Install
Create a virtual env and activate it.  
```python -m venv venv```  
```source venv/bin/activate```

Install requirements.  
```pip install -r requirements.txt```

Then install the gym.  
```pip install "gym[atari]"```

Accept licences.  
```pip install "gym[accept-rom-license, atari]"```
