import numpy as np

from bandit import EpsBandit
from utils.plots import plot_rewards


k = 10
iters = 1000

eps_0_rewards = np.zeros(iters)
eps_01_rewards = np.zeros(iters)
eps_1_rewards = np.zeros(iters)
eps_0_selection = np.zeros(k)
eps_01_selection = np.zeros(k)
eps_1_selection = np.zeros(k)

episodes = 1000

for i in range(episodes):
    eps_0 = EpsBandit(eps=0.0, k=k, iters=iters)
    eps_01 = EpsBandit(eps=0.01, k=k, iters=iters, mu=eps_0.mu.copy())
    eps_1 = EpsBandit(eps=0.1, k=k, iters=iters, mu=eps_0.mu.copy())

    # Run experiments
    eps_0.run()
    eps_01.run()
    eps_1.run()
    
    # Update long-term averages
    eps_0_rewards = (eps_0_rewards
                    + ((eps_0.reward - eps_0_rewards)
                    / (i + 1)))
    eps_01_rewards = (eps_01_rewards
                     + ((eps_01.reward - eps_01_rewards)
                     / (i + 1)))
    eps_1_rewards = (eps_1_rewards
                    + ((eps_1.reward - eps_1_rewards)
                    / (i + 1)))
    
    # Average actions per episode
    eps_0_selection = (eps_0_selection
                      + ((eps_0.k_n - eps_0_selection)
                      / (i + 1)))
    eps_01_selection = (eps_01_selection
                       + ((eps_01.k_n - eps_01_selection)
                       / (i + 1)))
    eps_1_selection = (eps_1_selection
                      + ((eps_1.k_n - eps_1_selection)
                      / (i + 1)))

plot_rewards(episodes, epsilon_0=eps_0_rewards,
    epsilon_01=eps_01_rewards, epsilon_1=eps_1_rewards)
