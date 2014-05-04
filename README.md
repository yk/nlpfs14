NLP 14 @ ETHZ
=============

All the files expect a folder called 'data' with the test data (1st subfolder should be duc2004) and a folder called 'rouge' with the rouge stuff in it (rouge pl script should be directly in there, as well as rouge's data folder).
These are currently in the .gitignore, since they are very large.

Usage
==
The ``parsetest.py`` file shows the usage of the nlpio module.
Only two functions are needed by a user, one being the ``loadDocumentsFromFile`` function, which loads all documents, including their manual models and peer suggestions, that are indexed by a file (here the ``testset.txt`` and the other function being the ``evaluateRouge`` function, where you put in the previously loaded documents as well as a list of your predictions (one per document) and receive a dictionary containing the recall, precision and F scores for the different rouge metrics along with their 95% confidence intervals. The test files also shows how to put in one of the peer suggestions as a prediction - maybe to establish a baseline.

Things to do:
==
* better file cleaning, maybe with some specialized tools
