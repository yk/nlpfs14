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

def setFeatureFromDict(features,dct,tag):
    try:
        features[dct[tag]] = 1
    except:
        pass

class CrfFeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        logging.info('extracting features')
        self.posTags = []
        self.nerTags = []
        self.depTags = []
        self.parseTags = []
        for doc in documents:
            for text in itt.chain([doc.ext['stanford']['article']],doc.ext['stanford']['models']):
                for sent in text['sentences']:
                    for word in sent['words']:
                        self.posTags.append(word[1]['PartOfSpeech'])
                        self.posTags.append(word[1]['NamedEntityTag'])
                        if 'dependency' in word[1]:
                            self.depTags.append(word[1]['dependency'][1])
                        if 'parsetreenode' in word[1]:
                            ptn = word[1]['parsetreenode']
                            while ptn is not None:
                                self.parseTags.append(ptn.name)
                                if not hasattr(ptn,'parent') or ptn.parent is None:
                                    break
                                ptn = ptn.parent
        self.posTags,self.nerTags,self.depTags,self.parseTags = map(lambda x : dict(zip(list(set(x)),itt.count())),[ self.posTags,self.nerTags,self.depTags,self.parseTags ])
        return self

    def transform(self,documents):
        logging.info('applying features')
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
                for rw in ['to','.',',','',';','?','!']:
                    if rw in labelWords:
                        labelWords.remove(rw)
                print labelWords
                totalIndex = -1
                depRoot = None
                for sentIndex,sent in enumerate(doc.ext['stanford']['article']['sentences']):
                    for wordIndex,word in enumerate(sent['words']):
                        totalIndex += 1
                        #if word[0] in ['.',',','to']:
                            #continue
                        labels.append(1 if word[0] in labelWords else 0)
                        posFeatures,nerFeatures,depFeatures,parseFeatures = map(lambda x : np.zeros(len(x),dtype=float),[self.posTags,self.nerTags,self.depTags,self.parseTags])
                        setFeatureFromDict(posFeatures,self.posTags,word[1]['PartOfSpeech'])
                        setFeatureFromDict(nerFeatures,self.nerTags,word[1]['NamedEntityTag'])
                        if 'dependency' in word[1]:
                            setFeatureFromDict(depFeatures,self.depTags,word[1]['dependency'][1])
                        if 'parsetreenode' in word[1]:
                            ptn = word[1]['parsetreenode']
                            while ptn is not None:
                                setFeatureFromDict(parseFeatures,self.parseTags,ptn.name)
                                if not hasattr(ptn,'parent') or ptn.parent is None:
                                    break
                                ptn = ptn.parent
                        features = np.concatenate((posFeatures,nerFeatures,depFeatures,parseFeatures))
                        nodes.append(features)
                        if 'dependency' in word[1]:
                            edges.append([totalIndex,totalIndex-wordIndex+word[1]['dependency'][0]])
                        else:
                            if depRoot is not None:
                                edges.append([depRoot,totalIndex])
                            depRoot = totalIndex
                #for i in range(len(nodes)-1):
                    #edges.append([i,i+1])
                doc.ext['crf'] = dict(nodes=np.array(nodes),edges=np.array(edges),labels=np.array(labels))
        return documents

class CrfEstimator(BaseEstimator):
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
            self.crf = OneSlackSSVM(model=model,C=1.0)

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


