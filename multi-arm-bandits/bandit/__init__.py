import numpy as np


class Bandit():
    """
    Bandit reinforcement learning algorithm that takes actions based
    upon precieved rewards. Generally, Gaussian distributions are
    utilized.

    Inputs:
        - eps   [double] Epsilon value. Strictly in [0,1].
        - k     [integer] Number of arms of bandit. Loosely number of
                          actions
        - iters [integer] Number of times to play the game
        - mu    [list] Distribution of rewards. Can be a string for
                       random generation.
    """

    def __init__(self, eps=0.0, k=1, iters=1000, mu='random'):
        # Number of iterations available
        self.iters = iters
        # Step Count
        self.n = 0
        # Number of arms
        self.k = k
        # Number of steps per arm
        self.k_n = np.zeros(k)
        # Mean reward over time
        self.mean_reward = 0
        # Individual award at each iteration
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Epsilon value for epsilon-greedy
        self.eps = eps

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            assert len(mu) == k
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
    
    def pull(self):
        # Greedy Bandit. Always take maximum reward
        a = np.argmax(self.k_reward)
        
        reward = np.random.normal(self.mu[1], 1)

        #update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        error = reward - self.mean_reward
        self.mean_reward = self.mean_reward + (error / self.n)

        # Update results for specific action
        k_error = reward - self.k_reward[a]
        self.k_reward[a] = self.k_reward[a] + (k_error / self.k_n[a])
    
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
    
    def reset(self):
        self.mean_reward = 0
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)


class EpsBandit(Bandit):
    """
    Extension of the Bandit reinforcement learning algorithm that
    utilizes an epsilon value to explore a portion of the time.
    """

    def pull(self):
        p = np.random.rand()
        if self.eps == 0.0 and self.n == 0:
            # Pick an action at radom from k
            a = np.random.choice(self.k)
        elif p < self.eps:
            a = np.random.choice(self.k)
        else:
            # Take a greedy action
            a = np.argmax(self.k_reward)
        
        reward = np.random.normal(self.mu[a], 1)

        #update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        error = reward - self.mean_reward
        self.mean_reward = (self.mean_reward + (error / self.n))

        # Update results for specific action
        k_error = reward - self.k_reward[a]
        self.k_reward[a] = self.k_reward[a] + (k_error / self.k_n[a])


class EpsDecayBandit(Bandit):
    """
    Extension of the Bandit reinforcement learning algorithm that
    utilizes a decay term on the epsilon greedy value to explore
    some of the time but then decrease the amount of exploration
    as time progresses.
    """

    def pull(self):
        p = np.random.rand()
        if p < 1 / (1 + self.n / self.k):
            # Pick an action at random from k
            a = np.random.choice(self.k)
        else:
            # Take a greedy action
            a = np.argmax(self.k_reward)
        
        reward = np.random.normal(self.mu[1], 1)

        #update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        error = (reward - self.mean_reward)
        self.mean_reward = self.mean_reward + (error / self.n)

        # Update results for specific action
        k_error = (reward - self.k_reward[a])
        self.k_reward[a] = self.k_reward[a] + (k_error / self.k_n[a])