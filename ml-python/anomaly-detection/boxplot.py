import seaborn as sns
import numpy as np 
import math
import matplotlib.pyplot as plt 

print("Generate random data")
rdata = np.random.randn(5000) * 20 + 20

sns.boxplot(data=rdata)