from ete2 import Tree
from pyparsing import nestedExpr
import nlpio
import re

def processNode(node):
    if type(node) == type('bla'):
        return node
    return node[0],[processNode(n) for n in node[1:]]

def renderNode(node):
    if node == ',':
        return 'COMMA'
    if type(node) == type('bla'):
        return node
    return "(" + " , ".join([renderNode(n) for n in node[1]]) + ")" + renderNode(node[0])


if __name__ == '__main__':
    sentence = 'Jim worked at a hotel. He was very happy with his life until Jane came into it. She was very arrogant and he disliked her. Still today, he\'s not very happy.'
    sentence = sentence.replace('.',' and ')
    if sentence.endswith(' and '):
        sentence = sentence[:-len(' and ')] + '.'
    parsd = nlpio.stanfordParse(sentence)
    for sent in parsd['sentences']:
        print sent['parsetree']
        print nestedExpr().parseString(sent['parsetree']).asList()[0]
        print processNode(nestedExpr().parseString(sent['parsetree']).asList()[0])
        print renderNode(processNode(nestedExpr().parseString(sent['parsetree']).asList()[0]))
        print Tree(renderNode(processNode(nestedExpr().parseString(sent['parsetree']).asList()[0])) + ";",format=1).get_ascii()
