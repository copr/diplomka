from scipy.stats import bernoulli

sampleSize = 2000
theta = 0.2

def generateSample(t, s):
    return bernoulli.rvs(t, size=s)

data = generateSample(theta, sampleSize)


theta_est = sum(data)/len(data)

print(theta_est)

