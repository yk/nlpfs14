from nlpio import *

import numpy as np
import logging
import random

from sklearn.base import BaseEstimator,TransformerMixin

class RougeScorer(object):
    #variant can be any rouge variant available in the output
    #metric should be one of P,R,F
    def __init__(self,variant='ROUGE-1',metric='F'):
        self.variant = variant
        self.metric = metric

    def __call__(self,estimator,documents,predictions=None):
        if predictions is None:
            predictions = estimator.predict(documents)
        results = evaluateRouge(documents,predictions)
        return results[self.variant][self.metric][0]

class HeadlineEstimator(BaseEstimator):
    '''Estimates, for a given document, its headline'''
    def __init__(self):
        pass

    def fit(self, documents, y=None): #we generate the headlines directly from the documents, so we don't need a "y"
        return self

    def predict(self, documents):
        return [doc.ext['sentences'][0] for doc in documents] #for now this just predicts the first sentence

