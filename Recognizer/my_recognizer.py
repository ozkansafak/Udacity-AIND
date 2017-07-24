import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list) as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # TODO implement the recognizer
    # return probabilities, guesses
    
    # iterate over test_set

    for idx in range(test_set.num_items):
        seq, seq_lengths = test_set.get_item_Xlengths(idx)
                
        logL_dict = {} # for ewach test_word prodiuce a dictionary logL = {'word': log-likelihood}
        
        # iterate over trained models
        for word, model in models.items():
            try:
                # calculate log-likelihood scores against the model of each trained "word"
                logL_dict[word] = model.score(seq, seq_lengths)
            except:
                # eliminate non-viable models from consideration.
                logL_dict[word] = -np.inf
                continue
        
        probabilities.append(logL_dict)
        # and pick the "key" that gives the max "value" in logL_dict.
        guesses.append(max(logL_dict, key=lambda x: logL_dict[x]))

    return probabilities, guesses
        
    
    
    
    
    
    
    
    