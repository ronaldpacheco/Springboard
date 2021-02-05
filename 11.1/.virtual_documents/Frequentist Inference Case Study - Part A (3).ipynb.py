from scipy.stats import norm
from scipy.stats import t
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt





seed(47)
# draw five samples here



# Calculate and print the mean here, hint: use np.mean()


















seed(47)
pop_heights = norm.rvs(172, 5, size=50000)


_ = plt.hist(pop_heights, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in entire town population')
_ = plt.axvline(172, color='r')
_ = plt.axvline(172+5, color='r', linestyle='--')
_ = plt.axvline(172-5, color='r', linestyle='--')
_ = plt.axvline(172+10, color='r', linestyle='-.')
_ = plt.axvline(172-10, color='r', linestyle='-.')


def townsfolk_sampler(n):
    return np.random.choice(pop_heights, n)


seed(47)
daily_sample1 = townsfolk_sampler(10)


_ = plt.hist(daily_sample1, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')


np.mean(daily_sample1)


daily_sample2 = townsfolk_sampler(10)


np.mean(daily_sample2)





seed(47)
# take your samples here












seed(47)
# calculate daily means from the larger sample size here















seed(47)
# take your sample now

























