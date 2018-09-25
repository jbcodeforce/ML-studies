'''
Created on May 24, 2016

Build a document Classifier to assess if a document is a spam or not

@author: jeromeboyer
'''

import math
from nis import cat
import re
from sqlite3 import dbapi2 as sqlite


def getwords(doc):
    ''' Extract features from a text: features will be words. So breaks the given document
    into words by dividing by any character that is not a letter
    '''
    splitter=re.compile('\\W*') # separate by Word
    words=[s.lower() for s in splitter.split(doc) if len(s)>2 and len(s) < 20]
    return dict([(w,1) for w in words])

class Classifier:
    '''
    '''
    def __init__(self,getfeatures,filename=None):
        # feature count looks like {'a_feature': {'classA':3,'classB':7}, ...
        self.featurecount={}
        # count how many time a class was used during training, # of document processed so far.
        # It is used for probability computation
        # the sum of classcount = nb of items processed overall.
        self.classcount={}
        self.getfeatures=getfeatures
        # Used to control the assignment to a category.
        self.thresholds = {}

    # Increase the count of a feature within a class
    def incFeature(self,f,c):
        # feature count looks like {'a_feature': {'classA':3,'classB':7}, ...
        self.featurecount.setdefault(f,{})
        self.featurecount[f].setdefault(c,0)
        self.featurecount[f][c]+=1

    # Increase the number of time a class was trained
    def increaseClassOccurence(self,c):
        self.classcount.setdefault(c,0)
        self.classcount[c]+=1

    # Return the number of time a feature has appeared in a class
    def featureCount(self,f,c):
        if f in self.featurecount and c in self.featurecount[f]:
            return float(self.featurecount[f][c])
        return 0.0

    # The number of items in a class
    def getClassCount(self,c):
        if c in self.classcount:
            return float(self.classcount[c])
        return 0.0

    def getTotalItems(self):
        return sum(self.classcount.values())

    def getClasses(self):
        return self.classcount.keys()

    def setThreshold(self,cat,t):
        self.thresholds[cat]=t

    def getThreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat];


    def featureProbability(self,f,c):
        ''' Compute the conditional probability of the feature f being part of the class c
        It is # of time f is in c / total # of doc in c
        It is called the conditional probability = P(F ! C)
        '''
        if self.getClassCount(c) == 0: return 0.0
        return self.featureCount(f, c)/self.getClassCount(c)


    def weightedProbability(self,f,c,fp,weigth=1.0, assumedProb=0.5):
        ''' Use assumed probability to avoid misclassifying a feature because at the beginning
        it is encountered in wrong category.
        '''
        # current probability
        p=fp(f, c)
        # count the # of time f appears in all classes
        n=sum([self.featureCount(f, c2) for c2 in self.getClasses()])
        # weighted average
        return ((weigth * assumedProb) + n*p)/(weigth + n)


    def train(self,item,c):
        ''' Take an item and a class, break the item into a set of features. Then
        count the number of time the feature is in the class.
        '''
        featureSet =self.getfeatures(item)
        for f in featureSet:
            self.incFeature(f, c)
        self.increaseClassOccurence(c)

    def sampleTrain(self):
        self.train('Nobody own the water','good')
        self.train('the quick rabbit jumps the fences', 'good')
        self.train('buy pharmaceuticals now', 'bad')
        self.train('make quick money in the online casino', 'bad')
        self.train('the quick frog jumps in the pound, but is not afraid of human', 'good')



class NaiveBayes(Classifier):
    ''' This classifier is using Bayes probability law, but also assumes that the probabilities being
    combined are independant of each other: the P(word_A | doc1) is unrelated to P(word_B |doc1).
    This is a false assumption as casino and quick money are related
    '''

    def documentProbability(self,item,c):
        '''Probability to have the entire document in category c
        '''
        features=self.getfeatures(item)
        # multiple the P of each feature
        p=1
        for f in features:
            p*=self.weightedProbability(f, c,self.featureProbability)
        return p

    def probCategoryGivenDoc(self,item,c):
        ''' Probability of having a category given a document, From Bayes theorem
        it is P(Cat | doc) =  P(Doc | Cat) * P(Cat) / P(doc)
        '''
        # P(Cat) = probability that a random document is part of category C = # of doc in C / # total # of docs
        categoryProb=self.getClassCount(c)/self.getTotalItems()
        docProb=self.documentProbability(item, c)
        # there is no need to compute P(doc) as the goal is to compare results for each C
        # since P(doc) is the same, it becomes a constants, and as the result is to compare prob,
        # not to compute the real prob
        return docProb*categoryProb


    def classify(self,item,default=None):
        '''
        Compute the probability for the item being part of a class, determine the largest
        but keep it only if it exceeds the next class by more than its thresholds
        '''
        probs={}
        maxNumber=0.0
        # find class with the highest prob
        for cat in self.getClasses():
            probs[cat]=self.probCategoryGivenDoc(item,cat)
            if (probs[cat]) > maxNumber:
                maxNumber=probs[cat]
                best=cat
         # verify the probability exceeds the threshold*next best
        for cat in probs:
            if cat == best:continue
            if probs[cat]*self.getThreshold(best)>probs[best]:return default
        return best

'''
The Fisher method Calculates the probabilities of a category for each feature in the document,
then it combines those probabilities and tests to see if the set of probabilities is
more or less likely than a random set
'''

class FisherClassifier(Classifier):

    def categoryProbability(self,feature,category):
        '''
        P(Category| feature). Proba that a document is in the given category and that a particular
        feature is in that document.
        # of doc in this category with the feature f / total number of doc with this feature
        '''
        clf=self.featureProbability(feature, category)
        if clf == 0: return 0
        # frequency of this feature in all category
        freqsum=sum([self.featureProbability(feature, c) for c in self.getClasses()])
        return clf/freqsum

    def fisherProbability(self,item,c):
        ''' multiple the proba
        '''
        features=self.getfeatures(item)
        p=1
        for f in features:
            p*=(self.weightedProbability(f, c,self.categoryProbability))
        fscore=-2*math.log(p)
        return self.invchi2(fscore,len(features)*2)

    def invchi2(self,chi,df):
        m = chi/2.0
        sumOfTerms = term = math.exp(-m)
        for i in range(1,df//2):
            term*=m/i
            sumOfTerms+=term
        return min(sumOfTerms,1.0)

    def __init__(self,getfeatures):
        Classifier.__init__(self,getfeatures)
        self.minimums={}

    def setminimum(self,cat,minimum):
        self.minimums[cat]=minimum

    def getminimum(self,cat):
        if cat not in self.minimums: return 0
        return self.minimums[cat]

    def classify(self,item,default=None):
        ''' Classify the given item by using the fisher probability of the item
        being part of a category and take the best probability
        '''
        # Loop through looking for the best result
        best=default
        maximum=0.0
        for c in self.getClasses():
            p=self.fisherProbability(item,c)
            # Make sure it exceeds its minimum: this is to avoid to classify a doc in less accurate category
            # better to let a spam being good than a good doc to be classified as spam
            if p>self.getminimum(c) and p>maximum:
                best=c
                maximum=p
        return best
