import matplotlib.pyplot as plt


def plot_rewards(episodes, plot_size=(12,8), **kwargs):
    """
        Description: Plot the given reward data histories to see how
        each algorithm stacks up over time. Add a legend to the plot
        and appropriate label and title the plot.

        Inputs:
            - episodes [integer] number of times each algo. was run.
            - **kwargs [dict] key, value pair with the value being
                              the recorded rewards as a numpy array
    """
    plt.figure(figsize=plot_size)
    for key, value in kwargs.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " 
        + str(episodes) + " Episodes")
    plt.show()