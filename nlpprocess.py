import re
from nlpio import *
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

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

    def __init__(self,annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner']):
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

