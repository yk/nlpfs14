import re
from nlpio import *
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from pyparsing import nestedExpr
import itertools as itt

class SimpleTextCleaner(BaseEstimator,TransformerMixin):
    #TODO: make better
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        for doc in documents:
            doc.text = re.sub("`|'|\"","",doc.text)
            doc.text = re.sub("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.","\\1",doc.text)
        return documents

class SentenceSplitter(BaseEstimator,TransformerMixin):
    #TODO: make better
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        for doc in documents:
            if not 'sentences' in doc.ext:
                doc.ext['sentences'] = [s.strip() for s in doc.text.split('.') if s]
        return documents

class StanfordParser(BaseEstimator, TransformerMixin):

    def __init__(self,annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner','parse','dcoref']):
        self.annotators = annotators

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        logging.info('analyzing documents with corenlp')
        for i, doc in enumerate(documents):
            if not 'stanford' in doc.ext:
                article = stanfordParse(doc.text,annotators=self.annotators)
                models = [stanfordParse(model,annotators=self.annotators) for model in doc.models]
                doc.ext['stanford'] = dict(article=article,models=models)
        return documents

class TreeNode(object):
    def __init__(self,name,children,index):
        self.name = name
        self.children = children
        self.index = index

    def findNode(self,word):
        if self.children[0] == word:
            return self
        if self.isLeaf():
            return None
        for child in self.children:
            wn = child.findNode(word)
            if wn is not None:
                return wn
        return None

    def isLeaf(self):
        return type(self.children[0]) == type('bla')

    def getValue(self):
        return self.children[0]

def createTreeNode(node,i=[0],index=dict()):
    if type(node) == type('bla'):
        return node
    ii = i[0]
    i[0] = ii+1
    tn = TreeNode(node[0],[createTreeNode(n,i=i,index=index) for n in node[1:]],ii)
    for child in tn.children:
        if type(child) != type('bla'):
            child.parent = tn
    index[ii] = tn
    return tn

def parseTreeString(treeString):
    d = dict()
    tree = createTreeNode(nestedExpr().parseString(treeString).asList()[0],i=[0],index=d)
    return tree,d

class StanfordTransformer(BaseEstimator,TransformerMixin):
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        for doc in documents:
            for text in itt.chain([doc.ext['stanford']['article']],doc.ext['stanford']['models']):
                for sent in text['sentences']:
                    if type(sent['parsetree']) == type('bla'):
                        sent['parsetree'],sent['parsetreeindex'] = parseTreeString(sent['parsetree'])
                    for i,word in enumerate(sent['words']):
                        wn = sent['parsetree'].findNode(word[0])
                        if wn is not None:
                            word[1]['parsetreenode'] = wn
        return documents

