import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 10
p = 0.5

r_values = np.arange(n + 1)

pmf_values = binom.pmf(r_values, n, p)

plt.figure(figsize=(8, 5))
plt.bar(r_values, pmf_values, color='skyblue', edgecolor='black')

plt.xlabel("Number of Successes (r)")
plt.ylabel("Probability Mass Function (PMF)")
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xticks(r_values)

plt.show()