import itertools as itt
import logging
import numpy as np
from nlpio import *
from nlplearn import *
from nlpprocess import *
from nlpfeature import *
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM

class ManualFeatureExtractor(BaseEstimator,TransformerMixin):
    """
    TODO: need the following
    - features on POS tags
    - edges of CRF
    - positional features?
    - ?
    """
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        logging.info('extracting crf features')
        self.wordFeatures = [WordTagFeature('PartOfSpeech'),WordTagFeature('NamedEntityTag')]
        for doc in documents:
            for text in itt.chain([doc.ext['stanford']['article']],doc.ext['stanford']['models']):
                for sent in text['sentences']:
                    for word in sent['words']:
                        for feature in self.wordFeatures:
                            feature.fit(word)
        return self

    def transform(self,documents):
        logging.info('applying crf features')
        for doc in documents:
            if not 'crf' in doc.ext:
                nodes = []
                edges = []
                labels = []
                labelWords = []
                for model in doc.ext['stanford']['models']:
                    for sent in model['sentences']:
                        for word in sent['words']:
                            labelWords.append(word[0])
                labelWords = set(labelWords)
                numFeatures = setOffsets(self.wordFeatures)
                for sent in doc.ext['stanford']['article']['sentences']:
                    for word in sent['words']:
                        labels.append(1 if word[0] in labelWords else 0)
                        features = np.zeros(numFeatures)
                        for feature in self.wordFeatures:
                            feature.extractAndSet(features,word)
                        nodes.append(features)
                for i in range(len(nodes)-1):
                    edges.append([i,i+1])
                doc.ext['crf'] = dict(nodes=np.array(nodes),edges=np.array(edges),labels=np.array(labels))
        return documents

class ManualEstimator(BaseEstimator):
    '''Wrapper around the crf stuff'''
    def __init__(self):
        pass

    def _buildCrf(self,documents):
        X = [(doc.ext['crf']['nodes'],doc.ext['crf']['edges']) for doc in documents]
        Y = [doc.ext['crf']['labels'] for doc in documents]
        return X,Y

    def _initCrf(self,documents):
        if not hasattr(self,'crf'):
            nfeatures = documents[0].ext['crf']['nodes'].shape[1]
            model = GraphCRF(n_states=2,n_features=nfeatures,directed=True)
            self.crf = OneSlackSSVM(model=model,C=self.C)

    def fit(self, documents, y=None): 
        logging.info("fitting crf")
        self._initCrf(documents)
        X,Y = self._buildCrf(documents)
        self.crf.fit(X,Y)
        return self

    def predict(self, documents):
        logging.info("predicting with crf")
        X,Y = self._buildCrf(documents)
        Z = self.crf.predict(X)
        logging.info("assembling final sentences")
        preds = []
        for z,doc in zip(Z,documents):
            pred = []
            i = 0
            for sent in doc.ext['stanford']['article']['sentences']:
                for word in sent['words']:
                    if z[i] == 1:
                        pred.append(word[0])
                    i += 1
            preds.append(" ".join(pred))
        return preds


