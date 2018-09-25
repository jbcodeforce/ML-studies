'''
Created on Jul 15, 2016

@author: jeromeboyer
'''
import unittest

import docclassifier
import pandas as pd


class Test(unittest.TestCase):


    def loadTrainingSet(self):
        trainingSet =pd.read_csv('../../data/device-trainSet.csv',header=None)
        return trainingSet

    def testFisherClassifier(self):
        df=self.loadTrainingSet()
        labels=df.iloc[:,1].values
        X=df.iloc[:,0].values
        cl=docclassifier.FisherClassifier(docclassifier.getwords)
        for x,y in zip(X,labels):
            print("Train "+x+" "+y)
            cl.train(x,y)
        assert cl.classify("my iphone did not keep charge", default="unknown") == 'battery'
        assert cl.classify("my ipad battery dies", default="unknown") == 'battery'
        assert cl.classify("the batteries I bought are excellent", default="unknown") == 'other'
        assert cl.classify("my car power is running fine", default="unknown") == 'other'




if __name__ == "__main__":
    print(" #######################################################")
    print(" Document Fisher Classifier CLASSIFIER Testing")
    unittest.main()

    print(" Done...")
    print(" #######################################################")
