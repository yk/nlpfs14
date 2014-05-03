import xml.etree.ElementTree as ET
import re

class Document(object):
    def __init__(self,name,path,modelPath='data/duc2004/eval/models/1/',peerPath='data/duc2004/eval/peers/1/'):
        self.name = name
        self.path = path
        self.fileName = self.path + self.name
        self.text = parseDocFile(self.fileName)
        self.modelPath = modelPath
        self.peerPath = peerPath

def parseDocFile(fileName):
    root = ET.parse(fileName).getroot()
    texts = root.findall('TEXT')
    if len(texts) != 1:
        raise Exception('File does not have a single text node')
    text = texts[0].text.replace('\n','')
    text = re.sub("`|'|\"","",text)
    text = re.sub("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.","\\1",text)
    return text

def parseSimpleFile(fileName):
    with open(fileName) as f:
        return f.read().replace('\n','')

def loadDocuments(fileNames,modelPath,peerPath):
    documents = []
    for fileName in fileNames:
        lis = fileName.rfind('/') + 1
        path,name = '',fileName
        if lis > 0:
            path,name = fileName[:lis],fileName[lis:]
        documents.append(Document(name,path,modelPath,peerPath))
    return documents

def loadDocumentsFromFile(fileName,modelPath='data/duc2004/eval/models/1/',peerPath='data/duc2004/eval/peers/1/'):
    with open(fileName) as f:
        lines = f.readlines()
        return loadDocuments(lines,modelPath,peerPath)


