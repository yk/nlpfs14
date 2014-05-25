import itertools as itt

class Feature(object):
    def __init__(self):
        self.tags = dict()
        self.offset = 0

    def addTag(self,name):
        if name not in self.tags:
            self.tags[name] = len(self.tags)

    def setOffset(self,offset):
        self.offset = offset

    def setFeature(self,features,tag,value=1,additive=False):
        if tag in self.tags:
            features[self.offset + self.tags[tag]] = value if not additive else value + features[self.offset + self.tags[tag]]
            
class SimpleFeature(Feature):
    def __init__(self):
        super(SimpleFeature,self).__init__()
        self._onlytag = "onlytag"

    def setFeature(self,features,value):
        super(SimpleFeature,self).setFeature(features,self._onlytag,value=value)

def setOffsets(features):
    offset = 0 
    for feature in features:
        feature.setOffset(offset)
        offset += len(feature.tags)
    return offset

class WordTagFeature(Feature):
    def __init__(self,tagName):
        super(WordTagFeature,self).__init__()
        self._tagName = tagName

    def fit(self,word):
        self.addTag(self.extract(word))

    def extract(self,word):
        return word[1][self._tagName]

    def extractAndSet(self,features,word):
        self.setFeature(features,self.extract(word))
