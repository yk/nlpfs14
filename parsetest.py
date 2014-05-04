import nlpio

if __name__ == '__main__':
    #print nlpio.parseDocFile('data/duc2004/docs/d30001t/APW19981016.0240')
    documents = nlpio.loadDocumentsFromFile('testset.txt')
    predictions = [doc.peers[0] for doc in documents] #using the given predictions
    print nlpio.evaluateRouge(documents,predictions)
    predictions = ['lorem ipsum dolor' for doc in documents] #using own predictions
    print nlpio.evaluateRouge(documents,predictions)
