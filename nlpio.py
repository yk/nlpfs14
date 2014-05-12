import xml.etree.ElementTree as ET
import re,glob,random,os,logging,threading
from corenlp import StanfordCoreNLP

class Document(object):
    def __init__(self,name,path,modelPath,peerPath):
        self.name = name
        self.path = path
        self.fileName = self.path + self.name
        self.text = parseDocFile(self.fileName)
        if not self.text:
            raise Exception('No text in document %s' % name)
        self.modelPath = modelPath
        self.peerPath = peerPath
        self.modelFileNames = glob.glob(self.modelPath + '*' + self.name)
        self.models = [parseSimpleFile(modelFileName) for modelFileName in self.modelFileNames]
        self.peerFileNames = glob.glob(self.peerPath + '*' + self.name)
        self.peers = [parseSimpleFile(peerFileName) for peerFileName in self.peerFileNames]

        #removing models and peers with empty content
        self.modelFileNames, self.models = zip(*[(mfn,mod) for mfn,mod in zip(self.modelFileNames,self.models) if mod])
        self.peerFileNames,self.peers = zip(*[(pfn,peer) for pfn,peer in zip(self.peerFileNames,self.peers) if peer])

        self.ext = dict() #this is intended as a dict where you can store stuff about this document during the computation, for example its text split into sentences

def parseDocFile(fileName):
    root = ET.parse(fileName).getroot()
    texts = root.findall('TEXT')
    if len(texts) != 1:
        raise Exception('File does not have a single text node')
    text = texts[0].text.replace('\n','')
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
        lines = [line.strip() for line in f.readlines()]
        return loadDocuments(lines,modelPath,peerPath)

def getId(name):
    return name.split('.')[-3]

def produceRougeInput(documents,predictionFileNames):
    lines = []
    lines.append('<ROUGE-EVAL version="1.0">')
    for i,document in enumerate(documents):
        lines.append('<EVAL ID="%d">' % (i+1))
        lines.append('<PEER-ROOT>')
        lines.append('.')
        lines.append('</PEER-ROOT>')
        lines.append('<MODEL-ROOT>')
        lines.append('.')
        lines.append('</MODEL-ROOT>')
        lines.append('<INPUT-FORMAT TYPE="SPL">')
        lines.append('</INPUT-FORMAT>')
        lines.append('<PEERS>')
        #for peer in document.peerFileNames:
        for peer in predictionFileNames:
            lines.append('<P ID="%s">'%getId(peer))
            lines.append(peer)
            lines.append('</P>')
        lines.append('</PEERS>')
        lines.append('<MODELS>')
        for model in document.modelFileNames:
            lines.append('<M ID="%s">'%getId(model))
            lines.append(model)
            lines.append('</M>')
        lines.append('</MODELS>')
        lines.append('</EVAL>')
    lines.append('</ROUGE-EVAL>')
    return "\n".join(lines)

def savePredictions(documents,predictions,predictionsPath):
    predictionFileNames = []
    for doc,pred in zip(documents,predictions):
        pfn = '%s/PRED.A.%s' % (predictionsPath,doc.name)
        with open(pfn,'w') as f:
            f.write(' %s ' % pred)
        predictionFileNames.append(pfn)
    return predictionFileNames

def evaluateRouge(documents,predictions,rougeBinary='rouge/ROUGE-1.5.5.pl',rougeData='rouge/data'):
    rint = random.randint(1,1000000000)
    tmpXmlFileName = 'tmp_%d.xml' % rint
    tmpOutFileName = 'tmp_%d.out' % rint
    tmpPredFolderName = 'tmp_pred_%d' % rint

    os.mkdir(tmpPredFolderName)

    predictionFileNames = savePredictions(documents,predictions,tmpPredFolderName)

    rougeInput = produceRougeInput(documents,predictionFileNames)
    with open(tmpXmlFileName,'w') as f:
        f.write(rougeInput)
    cmdString = '%s -e %s -a -c 95 -b 75 -m -n 4 -w 1.2 %s > %s' % (rougeBinary,rougeData,tmpXmlFileName,tmpOutFileName)
    os.system(cmdString)

    results = dict()
    with open(tmpOutFileName) as f:
        for line in f:
            if line.startswith('-----'):
                continue
            data = re.findall('A (ROUGE\\S+) Average_([RPF]): (\\d+\\.\\d+) \\(95%-conf.int. (\\d+\\.\\d+) - (\\d+\\.\\d+)\\)',line)[0]
            if data[0] not in results:
                results[data[0]] = dict()
            results[data[0]][data[1]] = (float(data[2]),(float(data[3]),float(data[4])))

    if True: #for debugging set to false, else true
        os.remove(tmpXmlFileName)
        os.remove(tmpOutFileName)
        for fn in glob.glob('%s/*' % tmpPredFolderName):
            os.remove(fn)
        os.rmdir(tmpPredFolderName)
    
    return results

stanford = None
stanfordLock = threading.Lock()
stanfordDefaultAnnotators =['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse','dcoref']

def stanfordParse(text, corenlpDir='corenlp/stanford-corenlp-full-2013-11-12/',annotators =stanfordDefaultAnnotators):
    global stanford
    if stanford is None:
        stanfordLock.acquire()
        try:
            if stanford is None:
                logging.info('loading stanford corenlp')
                with open('corenlp/nlp.properties','w') as f:
                    f.write("annotators = " + ", ".join(annotators))
                stanford = StanfordCoreNLP(corenlpDir,properties='nlp.properties')
                logging.info('done loading stanford corenlp')
        finally:
            stanfordLock.release()
    return stanford.raw_parse(text)
