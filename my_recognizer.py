import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    #implement the recognizer
    # Explore all the test sequences
    for X, length in test_set.get_all_Xlengths().values():
        # Dictionary to store the log prob for all the word models tested with the current set of frames (X,length)
        dict_prob = {}
        # Explore all the words of the database
        for word, word_model in models.items():
            try:
                dict_prob[word] = word_model.score(X,length)
            except:
                dict_prob[word] = float("-inf")
                continue
        # Append dictionary to the list of probabilities for the sets of testing frames
        probabilities.append(dict_prob)

        # Store dict in tuples with value as the first element
        inverse = [(value,key) for key,value in dict_prob.items()]
        # The best guess is the tuple having the highest value
        guesses.append(max(inverse)[1])
    return probabilities, guesses