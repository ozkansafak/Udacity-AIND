import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        '''     
        (Ref: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.392.9508&rep=rep1&type=pdf, Sec 3.5)
        BIC penalizes over-parameterized models.
        
        BIC = −2*ln(V) + k*ln(n)
        V = likelihood
        k = number of free parameters of HMM model
        n = size of dataset
        k*ln(n) is the penalty term
        '''
        BIC_lo = np.inf
        best_model = None

        try:
            for n_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(n_states)
                logL = hmm_model.score(self.X, self.lengths)
                k = n_states**2 + 2*hmm_model.n_features*n_states - 1
                BIC = (-2)*logL + k*np.log(self.X.shape[0])
                if BIC < BIC_lo:
                    BIC_lo = BIC
                    best_model = hmm_model
        except:
            pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    Extra Refs:
    https://discussions.udacity.com/t/discriminative-information-criterion-formula-words-vs-parameters/250880
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection based on DIC scores
        DIC_hi = -np.inf
        best_model = None
        
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n_states)
                word_seq, word_len = self.hwords[self.this_word]
                logL = hmm_model.score(word_seq, word_len)
            except Exception:
                # in case the model is not compatible
                # logL = -np.inf
                continue

            antiLogL = 0 # sum of log-likelihoods of all words other than self.this_word
            
            M = 0 # num of words
            for otherword in (set(self.words)-set([self.this_word])):
                otherword_seq, otherword_len = self.hwords[otherword]
                try:
                    logL_otherword = hmm_model.score(otherword_seq, otherword_len)
                    antiLogL += logL_otherword
                    M += 1

                except Exception:
                    continue
                    
            alpha = 1.0 # regularization parameter (= 1)
            DIC = logL - (alpha / (M - 1)) * antiLogL

            if DIC > DIC_hi:
                DIC_hi = DIC
                best_model = hmm_model

        return best_model
        
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection using CV
        
        best_score = -np.inf
        best_model = None
        n_splits = min(3,len(self.sequences))
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                logL = 0
                # If there are 2 or more sequences then use KFold() cross-validation
                if n_splits > 1:
                    split_method = KFold(n_splits=n_splits)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                         # Create train and test splits with KFold() and combine_sequences().
                         # store training set and accompanying sequence lengths in "self"
                         X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                         X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                         
                         hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                 random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                         logL += hmm_model.score(X_test, lengths_test)
                else:   
                    # There's only 1 sequence for "this_word". KFold() is not possible.
                    # Report the log-likelihood of the only sequence for "this_word"
                    hmm_model = self.base_model(n_states)
                    logL = hmm_model.score(self.X, self.lengths)
                
                avg_logL = logL / n_splits
                if  avg_logL > best_score:
                    best_score = avg_logL
                    best_model = self.base_model(n_states)        
            
            except:
                pass

        return best_model
        
        