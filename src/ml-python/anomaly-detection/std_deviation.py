'''
We can use standard deviation to identify outliers:
In statistical data distribution is approximately normal then 
about 68% of the data values lie within one standard deviation 
of the mean and about 95% are within two standard deviations, 
and about 99.7% lie within three standard deviations
'''
import numpy as np 

# prepare data

print("Generate random data")
data = np.random.randn(5000) * 20 + 20

def find_anomalies(data):
    anomalies = []
    # set the upper and lower limits to be at 3 sigma
    sigma = np.std(data)
    mean = np.mean(data)
    lower = mean - 3 * sigma
    upper = mean + 3 * sigma
    print("outliers will be below: " + str(lower) + " or higher than " + str(upper))
    for item in data:
        if item > upper or item < lower:
            anomalies.append(item)

    return anomalies

print(find_anomalies(data))