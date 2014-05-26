import re,os,pickle
from nlpio import *
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from pyparsing import nestedExpr
import itertools as itt

def isString(s):
    return isinstance(s,str) or isinstance(s,unicode)

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

class CachingStanfordParser(StanfordParser):

    def __init__(self,**kwargs):
        super(CachingStanfordParser,self).__init__(**kwargs)

    def transform(self, documents):
        logging.info('analyzing documents with corenlp')
        cachedir = 'cache'
        try:
            os.mkdir(cachedir)
        except:
            pass
        for i, doc in enumerate(documents):
            if not 'stanford' in doc.ext:
                cacheName = '%s/%s_%s' % (cachedir,doc.dirName,doc.name)
                if os.access(cacheName,os.F_OK):
                    logging.info('loading %s from cache' % doc.name)
                    with open(cacheName) as f:
                        doc.ext['stanford'] = pickle.load(f)
                else:
                    logging.info('evaluating %s with corenlp' % doc.name)
                    article = stanfordParse(doc.text,annotators=self.annotators)
                    models = [stanfordParse(model,annotators=self.annotators) for model in doc.models]
                    doc.ext['stanford'] = dict(article=article,models=models)
                    with open(cacheName,'w') as f:
                        pickle.dump(doc.ext['stanford'],f)
        return documents

class StanfordBatchParser(StanfordParser):

    def __init__(self,**kwargs):
        super(StanfordBatchParser,self).__init__(**kwargs)

    def transform(self, documents):
        logging.info('analyzing documents with corenlp')
        if not 'stanford' in documents[0].ext:
            parsed = stanfordBatchParse(documents)
            for doc,prs in zip(documents,parsed):
                doc.ext['stanford'] = dict(article=prs[0],models=prs[1])
        return documents

class TreeNode(object):
    def __init__(self,name,children,index):
        self.name = name
        self.children = children
        self.index = index

    def findNode(self,word,fromIndex=0):
        if self.isLeaf() and self.index >= fromIndex:
            if self.children[0] == word:
                return self
        else:
            for child in self.children:
                wn = child.findNode(word)
                if wn is not None:
                    return wn
        return None

    def isLeaf(self):
        return isString(self.children[0])

    def getValue(self):
        return self.children[0]

def createTreeNode(node,i=[0],index=dict()):
    if isString(node):
        return node
    ii = i[0]
    i[0] = ii+1
    tn = TreeNode(node[0],[createTreeNode(n,i=i,index=index) for n in node[1:]],ii)
    for child in tn.children:
        if not isString(child):
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
                    if isString(sent['parsetree']):
                        try:
                            sent['parsetree'],sent['parsetreeindex'] = parseTreeString(sent['parsetree'])
                        except Exception:
                            print "couldn't parse %s" % doc.name
                            print sent['parsetree']
                            raise
                    indx = 0
                    for i,word in enumerate(sent['words']):
                        wn = sent['parsetree'].findNode(word[0],fromIndex=indx)
                        if wn is not None:
                            word[1]['parsetreenode'] = wn
                            indx = wn.index + 1
                        dependency = None
                        for dep in sent['indexeddependencies']:
                            if int(dep[2].rsplit('-',1)[-1]) == i+1:
                                dependency = (int(dep[1].rsplit('-',1)[-1])-1,dep[0],[])
                                break
                        if dependency is not None:
                            word[1]['dependency'] = dependency
                    for i,word in enumerate(sent['words']):
                        if 'dependency' in word[1] and word[1]['dependency'][0] >= 0:
                            sent['words'][word[1]['dependency'][0]][1]['dependency'][2].append(i)
                    for word in sent['words']:
                        word[1]['corefout'] = []
                        word[1]['corefin'] = []
                if 'coref' in text:
                    for coref_outer in text['coref']:
                        for coref in coref_outer:
                            ssi = coref[0][1]
                            tsi = coref[1][1]
                            srcsent = text['sentences'][ssi]
                            trgsent = text['sentences'][tsi]
                            for swi in range(coref[0][3],coref[0][4]):
                                for twi in range(coref[1][3],coref[1][4]):
                                    srcsent['words'][swi][1]['corefout'].append((tsi,twi))
                                    trgsent['words'][twi][1]['corefin'].append((ssi,swi))
        return documents

