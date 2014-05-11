# NLP 14 @ ETHZ

All the files expect a folder called 'data' with the test data (1st subfolder should be duc2004) and a folder called 'rouge' with the rouge stuff in it (rouge pl script should be directly in there, as well as rouge's data folder).
It also requires a folder called ``stanford-corenlp-full-2013-11-12`` in the **~~root~~ corenlp** directory, which you can get from here: http://nlp.stanford.edu/software/stanford-corenlp-full-2013-11-12.zip
These are currently in the .gitignore, since they are very large.

## Requirements
sklearn pexpect unidecode xmltodict (can all be installed via pip)

## Usage
### I/O
The ``parsetest.py`` file shows the usage of the nlpio module.
Only two functions are needed by a user, one being the ``loadDocumentsFromFile`` function, which loads all documents, including their manual models and peer suggestions, that are indexed by a file (here the ``testset.txt`` and the other function being the ``evaluateRouge`` function, where you put in the previously loaded documents as well as a list of your predictions (one per document) and receive a dictionary containing the recall, precision and F scores for the different rouge metrics along with their 95% confidence intervals. The test files also shows how to put in one of the peer suggestions as a prediction - maybe to establish a baseline.
``stanfordtest.py`` shows how to use the Stanford CoreNLP library to do parsing and POS tagging magic. Loading up this library can take a while, but then it's kept as a singleton.
### Learning
The ```learntest.py``` file shows the usage of the nlplearn module.
The modules contains transformers and estimators to set up an sklearn pipeline and do a parameter grid search through cross-validation.
The branch ``basicsetup`` will always contain a 'clean slate' basic setup of the system with it always predicting the first sentence of a document as headline.
Fixes and improvements to the general setup should be cherry-picked or merged into this branch.

## Things to do:
* better file cleaning, maybe with some specialized tools

## Acknowledgements
The ``corenlp`` package has been authored by Hiroyoshi Komatsu and Johannes Castner (https://bitbucket.org/torotoki/corenlp-python)
