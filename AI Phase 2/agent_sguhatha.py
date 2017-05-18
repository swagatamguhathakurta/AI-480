from agents import Agent

import math
import random
import numpy
import csv

verdict = {}

def ClassSummary(dataset):
    converted = ClassMarking(dataset)
    verdict = {}
    for Val, occurance in converted.iteritems():
            verdict[Val] = Final(occurance)
    return verdict

def ClassMarking(dataset):
    converted = {}
    row = 0
    for i in range(len(dataset)):
            row = dataset[i]
            if (row[-1] not in converted):
                    converted[row[-1]] = []
            converted[row[-1]].append(row)
    return converted

def Final(dataset):
	verdict = [(Mean(attribute), StDev(attribute)) for attribute in zip(*dataset)]
	del verdict[-1]
	return verdict

def Mean(num):
	return sum(num)/float(len(num))
 
def StDev(num):
	avg = Mean(num)
	var = sum([pow(x-avg,2) for x in num])/float(len(num)-1)
	return math.sqrt(var)

def ProbCalc(x, Mean, StDev):
	exp = math.exp(-(math.pow(x-Mean,2)/(2*math.pow(StDev,2))))
	return (1 / (math.sqrt(2*math.pi) * StDev)) * exp
 
def ClassProbCalc(verdict, inputVal):
	prob = {}
	for Val, classSummaries in verdict.iteritems():
		prob[Val] = 1
		for i in range(len(classSummaries)):
			Mean, StDev = classSummaries[i]
			x = inputVal[i]
			prob[Val] *= ProbCalc(x, Mean, StDev)
	
	return prob
			
def Prediction(verdict, inputVal):
	prob = ClassProbCalc(verdict, inputVal)
	LabelBest, ProbBest = None, -1
	for Val, probability in prob.iteritems():
		if LabelBest is None or probability > ProbBest:
			ProbBest = probability
			LabelBest = Val
	return LabelBest


class Agent_sguhatha(Agent):

    def train(self, X, y):
        converted = {}
        global verdict
        data=numpy.column_stack((X,y))

        for i in range(len(X)):
            for j in range(len(X[1])+1): 
                w = data[i][j]
                if w == 'True':
                    data[i][j] = 10
                elif w == 'False':
                    data[i][j] = 5
                elif w == 'Excellent':
                    data[i][j] = 1
                elif w == 'Trash':
                    data[i][j] = 0
        print "X:", data
        data=data.astype(int)
        verdict = ClassSummary(data)

    def predict_prob_of_excellent(self, x):
        x=x.astype(int)
        for i in range(len(x)): 
            w = x[i]
            if w == 1:
                x[i] = 10
            elif w == 0:
                x[i] = 5
        result = Prediction(verdict, x)
        return result
