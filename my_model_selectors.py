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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #implement model selection based on BIC scores

        try:
            # Initializes best_score with the minimum value possible
            best_score = float("-inf")
            # Creates models with number of states between self.min_n_components and self.max_n_components
            for i in range(self.min_n_components, self.max_n_components+1):
                try:
                    # Creates the model and defines BIC score
                    hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(self.X,
                                                                                                   self.lengths)
                    #Calculates number of parameters
                    parameters = i*i + 2*i*len(self.X[0]) -1
                    #Calculates BIC score
                    score = (-2)*hmm_model.score(self.X, self.lengths)+parameters*np.log(len(self.X))
                except:
                    continue
                # If the score is better than the current one store the score and the number of states
                if best_score < score:
                    best_score = score
                    best_n_components = i

            if self.verbose:
                print("model selected for {}".format(self.this_word))
            return self.base_model(best_n_components)

        except:
            if self.verbose:
                print("failure on {}".format(self.this_word))
            return None

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores

        try:
            # Initializes best_score with the minimum value possible
            best_score = float("-inf")
            # Creates models with number of states between self.min_n_components and self.max_n_components
            for i in range(self.min_n_components, self.max_n_components+1):
                try:
                    # Creates the model and initializes values to calculate DIC score
                    hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X,
                                                                                               self.lengths)
                    likeLog = hmm_model.score(self.X, self.lengths)
                    antilikeLog = 0
                    total_scores = 0
                    #Stores all likelihoods for the other words in the set of words
                    for word in self.words:
                        if (word== self.this_word):
                            continue
                        X, length = self.hwords[word]
                        try:
                            antilikeLog += hmm_model.score(X,length)
                            total_scores+= 1
                        except:
                            continue
                    #Normalize log likelihood value
                    antilikeLog /= total_scores
                    #Define the score as the different between the log likelihood of the data of the words and the average of the sum of likelihoods for the rest of the words
                    score = likeLog - antilikeLog
                except:
                    continue
                # If the score is better than the current one store the score and the number of states
                if best_score < score:
                    best_score = score
                    best_n_components = i

            if self.verbose:
                print("model selected for {}".format(self.this_word))
            return self.base_model(best_n_components)

        except:
            if self.verbose:
                print("failure on {}".format(self.this_word))
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            #Initializes best_score with the minimum value possible
            best_score = float("-inf")
            #Creates models with number of states between self.min_n_components and self.max_n_components
            for i in range(self.min_n_components, self.max_n_components+1):
                #Splits self.sequences in folds
                split_method = KFold(n_splits=min(3,len(self.sequences)))
                #List to store the log likelihood of all the KFolds of a model for a given state
                logList =[]
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    #Combine sequences to form training data
                    training_data,length_train = combine_sequences(cv_train_idx, self.sequences)
                    #Combine the sequences to form the testing data
                    test_data, length_test = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        #Creates the model and append the score (log likelihood) associated to the specific KFold element
                        hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(training_data, length_train)
                        logList.append(hmm_model.score(test_data, length_test))
                    except:
                        continue
                if (len(logList)>0):
                    #Take the average of the scores for all the possible separation of sequences given by KFold
                    logL_avg = sum(logList)/len(logList)
                    #If the score is better than the current one store the score and the number of states
                    if best_score < logL_avg:
                        best_score = logL_avg
                        best_n_components = i
                else:
                    continue

            if self.verbose:
                print("model created for {}".format(self.this_word))
            return self.base_model(best_n_components)

        except:
            if self.verbose:
                print("failure on {}".format(self.this_word))
            return None

