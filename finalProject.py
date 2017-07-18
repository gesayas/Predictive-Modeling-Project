# Name: Gabriel Esayas

import random
import math

from collections import defaultdict

import pylab as plt

class Model:

	'''
	Initializes a model. iprobs is a list assumed to begin with
	p2 and going through to and including p10 (it will be assumed
	p1 = 1 and p11 = 0, as defined in the paper)
	'''
	def __init__(self, iprobs):
		self.iprobs = iprobs

	'''
	Provides a list of the probabilities that the model rings
	exactly i times for i in the range [1, 10]
	'''
	def getRingDistribution(self):
		ringDistribution = [(1 - self.iprobs[0])] # at i = 1, probability of stopping is 1-p2
												  # (and p2 should be the first thing in iprobs)
		runningProbability = self.iprobs[0] # just decided to ring twice
		for i in xrange(1, 9):
			ringDistribution.append(runningProbability*(1 - self.iprobs[i])) # probability we stop here is added
			runningProbability *= self.iprobs[i] # decided to ring again
		ringDistribution.append(runningProbability) # at this point, runningProbability is the
													# product of all probabilities in iprobs, which
													# just so happens to be the probability that we ring 10 times
		return ringDistribution

	'''
	Runs n trials of a millionaire or billionaire game (depending
	on the boolean argument) to determine how lucrative this model is.
	Returns the average total earnings of the simulation, as well as a
	dictionary of all possible earnings mapped to the number of times
	they were achieved
	'''
	def runNTimes(self, n, millionaire = True):
		earningsDict = defaultdict(int)
		totalEarnings = 0
		for i in xrange(n):
			Z = random.randint(1, 11) # the host chooses the secret number
			if Z == 1:
				earningsDict[0] += 1 # player got unlucky. Not possible to win anything
				continue
			possibleEarnings = 100.0 # the player always rings at least once to try to win something
			for p in xrange(len(self.iprobs)):
				stopHereProb = random.random()
				if stopHereProb > self.iprobs[p]:
					totalEarnings += possibleEarnings # decided to stop on our own, so we win what we currently have
					earningsDict[possibleEarnings] += 1
					break
				if Z > p + 2: # i.e. say p = 0, it means we just decided above to ring twice
							  # (iprobs starts with p2). so Z must be 3 or more to continue
					if millionaire:
						possibleEarnings += 100*(p + 2) # millionaire adds 100*current ring amount
					else:
						possibleEarnings *= 3 # billionaire triples your earnings
					if p == len(self.iprobs) - 1:
						totalEarnings += possibleEarnings # player got lucky. Just won the most amount of money
						earningsDict[possibleEarnings] += 1
						break
				else:
					earningsDict[0] += 1 # player rang too much and lost everything.
					break
		return totalEarnings/n, earningsDict

n = 100000

# Model 0 - Ring 10 times (the "correct" strategy for the billionaire game)

M0 = Model(9*[1.0])

dist0 = M0.getRingDistribution()
print dist0

results0Mill = M0.runNTimes(n)
print results0Mill

results0Bill = M0.runNTimes(n, False)
print results0Bill

# Model 1 - Flip a coin

M1 = Model(9*[0.5])

dist1 = M1.getRingDistribution()
print dist1

results1Mill = M1.runNTimes(n)
print results1Mill

results1Bill = M1.runNTimes(n, False)
print results1Bill

# Model 2 - Conditional thinking

iprobsM2 = []
for i in xrange(2, 11):
	iprobsM2.append((11.0 - i)/(12.0 - i))

M2 = Model(iprobsM2)

dist2 = M2.getRingDistribution()
print dist2

results2Mill = M2.runNTimes(n)
print results2Mill

results2Bill = M2.runNTimes(n, False)
print results2Bill

# Model 3 (Not in report) - Ring 7 times (the "correct" strategy for the millionaire game)

M3 = Model([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0])

dist3 = M3.getRingDistribution()
print dist3

results3Mill = M3.runNTimes(n)
print results3Mill

results3Bill = M3.runNTimes(n, False)
print results3Bill

# Model 4 - High Risk Ratio: Millionaire game

E1 = 1000.0/11 # normalizing factor (used in Model 5 as well)

iprobsM4 = []
for i in xrange(2, 11):
	iprobsM4.append(1 - math.exp(-200.0*i*(11 - i)/11/(i - 1)/E1))
print iprobsM4

M4 = Model(iprobsM4)

dist4 = M4.getRingDistribution()
print dist4

results4Mill = M4.runNTimes(n)
print results4Mill

results4Bill = M4.runNTimes(n, False)
print results4Bill

# Model 5 - Conservative Risk Ratio: Millionaire game

iprobsM5 = []
for i in xrange(2, 11):
	iprobsM5.append(1 - math.exp(-200.0*i*(11 - i)/121*(11 - i)/(i - 1)/E1))
print iprobsM5

M5 = Model(iprobsM5)

dist5 = M5.getRingDistribution()
print dist5

results5Mill = M5.runNTimes(n)
print results5Mill

results5Bill = M5.runNTimes(n, False)
print results5Bill

# Model 6 - Revised Billionaire Model

iprobsM6 = []
for i in xrange(2, 11):
	iprobsM6.append(1 - math.exp(-3*1.9955471026787518/(i - 1)))
print iprobsM6

M6 = Model(iprobsM6)

dist6 = M6.getRingDistribution()
print dist6

results6Mill = M6.runNTimes(n)
print results6Mill

results6Bill = M6.runNTimes(n, False)
print results6Bill

# Model 7 - Actual Human Responses for the millionaire

humanResponse1 = [6.0, 2.0, 17.0, 13.0, 16.0, 4.0, 13.0, 2.0, 0.0, 3.0]

iprobsM7 = []
numLeft = 76.0 # keeps track of number of humans left to correctly calculate conditional probs
for i in xrange(9):
	iprobsM7.append((numLeft - humanResponse1[i])/numLeft)
	numLeft -= humanResponse1[i]
print iprobsM7

M7 = Model(iprobsM7)

dist7 = M7.getRingDistribution()
print dist7

results7Mill = M7.runNTimes(n)
print results7Mill

results7Bill = M7.runNTimes(n, False)
print results7Bill

# Model 8 - Actual Human Responses for the billionaire

humanResponse2 = [5.0, 2.0, 10.0, 19.0, 5.0, 12.0, 8.0, 1.0, 3.0, 11.0]

iprobsM8 = []
numLeft = 76.0
for i in xrange(9):
	iprobsM8.append((numLeft - humanResponse2[i])/numLeft)
	numLeft -= humanResponse2[i]
print iprobsM8

M8 = Model(iprobsM8)

dist8 = M8.getRingDistribution()
print dist8

results8Mill = M8.runNTimes(n)
print results8Mill

results8Bill = M8.runNTimes(n, False)
print results8Bill

i_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure()
plt.plot(i_values, dist0, 'y--', label = 'probability distribution of M0 (Ring 10 Times)')
plt.plot(i_values, dist1, 'm--', label = 'probability distribution of M1 (Fair Coin Flip)')
plt.plot(i_values, dist2, 'g--', label = 'probability distribution of M2 (Conditional Thinking)')
plt.plot(i_values, dist3, 'c--', label = 'probability distribution of M3 (Ring 7 Times)')
plt.plot(i_values, dist4, 'r--', label = 'probability distribution of M4 With High Risk Ratio')
plt.plot(i_values, dist5, 'b--', label = 'probability distribution of M5 With Conservative Risk Ratio')
plt.plot(i_values, dist7, 'k--', label = 'probability distribution of Human Responses')
plt.xlabel('Number of Times Decided to Ring (i)')
plt.ylabel('Probability of Deciding to Ring i Times')
plt.title('Probability Distributions for the Millionaire Game Compared With Distribution of Human Responses')
plt.legend(loc = 'upper left')
plt.show()


plt.figure()
plt.plot(i_values, dist0, 'y--', label = 'probability distribution of M0 (Ring 10 Times)')
plt.plot(i_values, dist1, 'm--', label = 'probability distribution of M1 (Fair Coin Flip)')
plt.plot(i_values, dist2, 'g--', label = 'probability distribution of M2 (Conditional Thinking)')
plt.plot(i_values, dist3, 'c--', label = 'probability distribution of M3 (Rings 7 Times)')
plt.plot(i_values, dist6, 'r--', label = 'probability distribution of M6 (Revised Billionaire Model)')
plt.plot(i_values, dist8, 'k--', label = 'probability distribution of Human Responses')
plt.xlabel('Number of Times Decided to Ring (i)')
plt.ylabel('Probability of Deciding to Ring i Times')
plt.title('Probability Distributions for the Billionaire Game Compared With Distribution of Human Responses')
plt.legend(loc = 'upper left')
plt.show()

objects = ["Ring 10 Times", "Flip A Coin", "Conditional", "Ring 7 Times", "High Risk", "Conservative Risk", "Billionaire Model", "Millionaire Response", "Billionaire Response"]
yMill = [results0Mill[0], results1Mill[0], results2Mill[0], results3Mill[0], results4Mill[0], results5Mill[0], results6Mill[0], results7Mill[0], results8Mill[0]]

plt.figure()
plt.bar(range(9), yMill, align='center')
plt.xlabel('Model')
plt.xticks(range(9), objects)
plt.ylabel('Average Earnings')
plt.title('Millionaire Game Performance After 100000 simulations')
plt.show()

yBill = [results0Bill[0], results1Bill[0], results2Bill[0], results3Bill[0], results4Bill[0], results5Bill[0], results6Bill[0], results7Bill[0], results8Bill[0]]

plt.figure()
plt.bar(range(9), yBill, align='center')
plt.xlabel('Model')
plt.xticks(range(9), objects)
plt.ylabel('Average Earnings')
plt.title('Billionaire Game Performance After 100000 simulations')
plt.show()