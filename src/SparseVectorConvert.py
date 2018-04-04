# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:08:35 2018

@author: cs
"""

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter


#import numpy as np

class MovieReviews:
    
    def readText(filename):
        DF = pd.read_csv(filename, sep = "\n", header = None)
        DF.columns = ['Reviews']
        return DF
    
 
    def splitDataframe(pandasDataFrame):
        train, test = train_test_split(pandasDataFrame, test_size = 0.15)
        return train, test
    
    def mergeDataFrame(pandasDataFrame1, pandasDataFrame2, pos, neg):
        DF1 = pandasDataFrame1.assign(label = pos)
        DF2 = pandasDataFrame2.assign(label = neg)
        df_new = pd.concat([DF1, DF2])
        return df_new
        
    
    def getConfusionMatrix(pandasDataFrame, predictionList):
        conf_matrix = metrics.confusion_matrix(pandasDataFrame.label, predictionList)
        return conf_matrix    
        
    def tokenizeWords(sent):
        word_tokens = word_tokenize(sent)
        return word_tokens
    
    
    def convertCounter(dataframe):
        wordlist = []
        for index, row in dataframe.iteritems():
            wordlist.extend(row)
        cnt = Counter(wordlist)
        #print(cnt.keys())
        return cnt

    def getIndexing(dictionary):
        words = dictionary.keys()
        new_words = list(enumerate(words))
        a = dict(new_words)
        return a
    
            
    def convertToSparseMatrix(dataframe, dictionary, label, filename):
        res = dict((v, k) for k, v in dictionary.items())
        f = open(filename,"w+")
        for index, row in dataframe.iteritems():
            a_dict = dict()
            countsen = Counter(row)
            for words, freq in countsen.items():
                a = res.get(words, -1)
                a_dict[a] = freq
            f.write('%d' % label)
            for key, values in sorted(a_dict.items()):
                f.write(' ')
                f.write('%d' % key)
                f.write(":")
                f.write('%d' % values)
            f.write("\r\n")
            
    def addToSparseMatrix(dataframe, dictionary, label, filename):
        res = dict((v, k) for k, v in dictionary.items())
        f = open(filename,"a")
        for index, row in dataframe.iteritems():
            a_dict = dict()
            countsen = Counter(row)
            for words, freq in countsen.items():
                a = res.get(words, -1)
                a_dict[a] = freq
            f.write('%d' % label)
            for key, values in sorted(a_dict.items()):
                f.write(' ')
                f.write('%d' % key)
                f.write(":")
                f.write('%d' % values)
            f.write("\r\n")
              
                    
                
    
if __name__ == '__main__':
    RD = MovieReviews
    posReviewDF = RD.readText("rt-polarity.pos")
    negReviewDF = RD.readText("rt-polarity.neg")
    #print (negReviewDF) 
    new_posReviewDF = posReviewDF['Reviews'].apply(RD.tokenizeWords)
    new_negReviewDF = negReviewDF['Reviews'].apply(RD.tokenizeWords)
    #print(new_negReviewDF)
    merged_df = pd.concat([new_posReviewDF, new_negReviewDF])
    merged_df.reset_index()
    print(merged_df)
    fullDict = RD.convertCounter(merged_df)
    fullDictIndexing = RD.getIndexing(fullDict)

     
    posReviewDFTrain1, posReviewDFTest = RD.splitDataframe(new_posReviewDF)
    negReviewDFTrain1, negReviewDFTest = RD.splitDataframe(new_negReviewDF)
    posReviewDFTrain, posReviewDFVal = RD.splitDataframe(posReviewDFTrain1)
    negReviewDFTrain, negReviewDFVal = RD.splitDataframe(negReviewDFTrain1)
    print(negReviewDFTrain.shape)  
    print(posReviewDFTrain.shape)
  
    sparseNegReview = RD.convertTOSparseMatrix(negReviewDFTrain , fullDictIndexing, 0)
    
    sparsePosReview = RD.convertTOSparseMatrix(posReviewDFTrain , fullDictIndexing, 1)
    
    merged_training_data = pd.concat([sparseNegReview, sparsePosReview])
    
    RD.convertToSparseMatrix(posReviewDFTrain , fullDictIndexing, 1, "reviewTrainData.txt")
    RD.addToSparseMatrix(negReviewDFTrain, fullDictIndexing, 0, "reviewTrainData.txt")
    RD.convertToSparseMatrix(posReviewDFVal , fullDictIndexing, 1, "reviewValData.txt")
    RD.addToSparseMatrix(negReviewDFVal, fullDictIndexing, 0, "reviewValData.txt")
    RD.convertToSparseMatrix(posReviewDFTest , fullDictIndexing, 1, "reviewTestData.txt")
    RD.addToSparseMatrix(negReviewDFTest, fullDictIndexing, 0, "reviewTestData.txt")
    
    