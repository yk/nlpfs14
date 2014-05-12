import itertools as itt
import logging
import numpy as np
from nlpio import *
from nlplearn import *
from nlpprocess import *
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class CrfFeatureExtractor(BaseEstimator,TransformerMixin):
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
        self.posTags = []
        self.nerTags = []
        for doc in documents:
            for text in itt.chain([doc.ext['stanford']['article']],doc.ext['stanford']['models']):
                for sent in text['sentences']:
                    for word in sent['words']:
                        self.posTags.append(word[1]['PartOfSpeech'])
                        self.nerTags.append(word[1]['NamedEntityTag'])
        self.posTags,self.nerTags = itt.imap(lambda x: dict(zip(list(set(x)),itt.count())),[self.posTags,self.nerTags])
        logging.info('extracted %d POS tags' % len(self.posTags.items()))
        logging.info('extracted %d NER tags' % len(self.nerTags.items()))
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
                for sent in doc.ext['stanford']['article']['sentences']:
                    for word in sent['words']:
                        labels.append(1 if word[0] in labelWords else 0)
                        features = np.zeros(len(self.posTags) + len(self.nerTags))
                        features[self.posTags[word[1]['PartOfSpeech']]] = 1
                        features[self.nerTags[word[1]['NamedEntityTag']]] = 1
                        nodes.append(features)
                for i in range(len(nodes)-1):
                    edges.append([i,i+1])
                doc.ext['crf'] = dict(nodes=np.array(nodes),edges=np.array(edges),labels=np.array(labels))
        return documents
