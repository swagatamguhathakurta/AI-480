'''
agent_sguhatha implementation
'''

#importing the agent class and sklearn libraries.
from agents import Agent
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

#function to train,test and then ivaluate the wealth of the classifer.
def best(classifier, X_train, y_train, X_val, y_val,value ,price_trials):

        #Setting classifier as based on the input
        if classifier == "BernoulliNB": 
            clf=BernoulliNB()
        elif classifier == "LogisticRegression":
            clf=LogisticRegression()
        elif classifier == "SVC":
            clf=SVC(kernel='poly',degree=4,probability=True,random_state=0)

        #Train the classifier 
        clf.fit(X_train, y_train)

        index={}
        #Mark the index array based on excellent or trash
        if clf.classes_[0] == 'Excellent':
            index = 0
        else:
            index = 1

        wealths=0
        
        #run the loop for number of products 
        for i in range(1000):
            excellent = (y_val[i] == 'Excellent')

            #Find the probability for excellence of the validation data.
            prob = clf.predict_proba(X_val[i])[0][index]
            for pt in range(price_trials):
                #Find price of the product.
                price = ((2*pt+1)*value)/(2*price_trials)
                #set the wealth of the classifier based on buying or not.
                if value*prob > price:
                    wealths -= price
                    if excellent:
                        wealths += value
        return wealths 

class Agent_sguhatha(Agent):
    def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
        b=[0,0,0]
        m=0
        o=0
        value=1000
        price_trials=10

        classifiers = ["BernoulliNB","LogisticRegression","SVC"]
        
        #call the built function recursively for all the classifiers.
        for i in range(3): 
                b[i]=best(classifiers[i], X_train, y_train, X_val, y_val,value ,price_trials)
                if m<b[i]:
                        m=b[i]
                        o=i

        #Decide and return the welthiest classifier.
        if o == 0:
            clf=BernoulliNB()
        elif o == 1:
            clf=LogisticRegression()
        elif o == 2:
            clf=SVC(kernel='poly',degree=4,probability=True,random_state=0)
        return clf
