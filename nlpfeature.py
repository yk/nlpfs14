class Feature(object):
    def __init__(self,name):
        self.name = name
        self.tags = dict()
        self.offset = 0

    def addTag(self,name):
        if name not in self.tags:
            self.tags[name] = len(self.tags)

    def setOffset(self,offset):
        self.offset = offset

    def setFeature(self,tag,features):
        if tag in self.tags:
            features[self.offset + self.tags[tag]] = 1
            
    def setFeatures(self,tags,features):
        for tag in tags:
            self.setFeature(tag,features)

    def extract(self,datum):
        pass

def setOffsets(features):
    offset = 0 
    for feature in features:
        feature.setOffset(offset)
        offset += len(feature.tags)
    return offset
