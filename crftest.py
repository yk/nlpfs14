from nlpio import *
from nlplearn import *
from nlpprocess import *
from nlpcrf import *

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    documents = loadDocumentsFromFile('testset_large.txt')


    pipeline = Pipeline([
        ('clean',SimpleTextCleaner()),
        ('sentence',SentenceSplitter()),
        ('stanford',CachingStanfordParser()),
        ('stanfordtransform',StanfordTransformer()),
        ('crffeatures',CrfFeatureExtractor()),
        ('crf',CrfEstimator()),
        ])
    
    trainDocs,testDocs = train_test_split(documents,test_size=0.2)
    scorer = RougeScorer()
    #parameters for cross-validation grid search go here
    """
    parameters = {
            
            }


    grid_search = GridSearchCV(pipeline, parameters, scoring=scorer, cv=(5 if parameters else 2), n_jobs=1, refit=True, verbose=3)


    grid_search.fit(trainDocs)

    print("Best score: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("score on test set: %f" % scorer(grid_search,testDocs))
    """

    pipeline.fit(trainDocs)
    print pipeline.predict(testDocs)
    #print("score on test set: %f" % scorer(pipeline,testDocs))
    
