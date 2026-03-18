'''
Created on May 20, 2016

@author: jeromeboyer
'''

'''
The Naive Bayes algorithm computes the probability for each attribute to belong to
each class. It is using a supervised learning approach. It assumes that the 
probability of each attribute belonging to a given class is independent of all 
other attributes.

The naive bayes model is comprised of a summary of the data in the training dataset. 
This summary is then used when making predictions. The summary of the training data
 collected involves the mean and the standard deviation for each attribute, by class value.
'''

import math


class NaiveBayesClassifier:
    
    def separateByClass(self,dataset):
        ''' Returns a map of class values from the data set
        ''' 
        separatedSet={}
        for i in range(len(dataset)):
            classe=dataset[i][-1]
            if classe not in separatedSet:
                separatedSet[classe]=[]
            separatedSet[classe].append(dataset[i])
            pass
        return separatedSet
    
    
    def mean(self,numbers):
        return sum(numbers)/float(len(numbers))
    
    def stdDeviation(self,numbers):
        '''  The standard deviation describes the variation of spread of the data, 
        and it is used to characterize the expected spread of each attribute
         in the Gaussian distribution when calculating probabilities
        '''
        average=self.mean(numbers)
        # The standard deviation is calculated as the square root of the variance.
        variance=sum([pow(x - average,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)
              
    def summarize(self,dataset):
        ''' Keep mean and std deviation for each attributes
        '''
        summaries=[(self.mean(attribute),self.stdDeviation(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries
    
    
    def summarizeByClass(self,dataset):
        ''' separating the dataset into instances grouped by class. 
        return the (mean, standard Deviation) tuple of each attribute of the data set and per class
        '''
        byClass=self.separateByClass(dataset)
        summaries={}
        for classValue,instances in byClass.items():
            summaries[classValue]=self.summarize(instances)
        return summaries
        
    
    
    def calculateProbability(self,x,mean,standardDeviation):
        ''' Get the probability of x using the Gaussian distribution
        '''
        e=math.exp(-(math.pow(x-mean,2)/(2*math.pow(standardDeviation,2))))
        return (1 / (math.sqrt(2*math.pi) * standardDeviation)) * e
     
     
    def calculateClassProbabilities(self,summaries, record):
        ''' 
        ''' 
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                probabilities[classValue] *= self.calculateProbability(record[i], mean, stdev)
        return probabilities
   
    def predict(self,summaries,record):
        ''' From the feature set of the given record use the probability to 
        find the class this record most likely belongs to
        '''
        probabilities = self.calculateClassProbabilities(summaries, record)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel
        
    
    def getPredictions(self,summaries,dataset):
        ''' Given the dataset with no class assigned to it build
        the list of class prediction for each record of the dataset
        ''' 
        predictions = []
        for i in range(len(dataset)):
            result = self.predict(summaries, dataset[i])
            predictions.append(result)
        return predictions
    
    def getAccuracy(self,testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0