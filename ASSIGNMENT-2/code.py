import numpy as np
import random
import string 
from sklearn.tree import DecisionTreeClassifier

# Define functions for generating bigrams and creating multi-hot vectors

def create_bigrams(term, limit=None):
    # Get all bigrams
    bigrams = [''.join(bigram) for bigram in zip(term, term[1:])]
    # Remove duplicates and sort them
    bigrams = sorted(set(bigrams))
    # Make them into an immutable tuple and retain only the first few
    return tuple(bigrams)[:limit]

def generate_multi_hot(terms):
    all_bigrams = []
    term_bigrams_map = {}

    for term in terms:
        bigram_list = list(create_bigrams(term))
        term_bigrams_map[term] = bigram_list
        all_bigrams.extend(bigram_list)

    unique_bigrams = sorted(set(all_bigrams))
    unique_bigrams_len = len(unique_bigrams)

    global term_index_map
    term_index_map = {}
    terms_indices = []
    for idx in range(len(terms)):
        term_index_map[idx] = terms[idx]
        terms_indices.append(idx)

    multi_hot_matrix = []
    for term in terms:
        vector = [0] * unique_bigrams_len
        bigrams = term_bigrams_map[term]

        for i, bigram in enumerate(unique_bigrams):
            if bigram in bigrams:
                vector[i] = 1
        multi_hot_matrix.append(vector)

    return multi_hot_matrix, unique_bigrams, terms_indices

# Implementing the required functions for training and prediction

def my_fit(terms):
    global unique_bigrams, x_train_matrix
    x_train_matrix, unique_bigrams, terms_indices = generate_multi_hot(terms)
    y_train_labels = terms_indices

    clf = DecisionTreeClassifier(random_state=0, criterion='gini')
    clf.fit(x_train_matrix, y_train_labels)

    return clf

def my_predict(model, bigrams):
    input_vector = [0] * len(unique_bigrams)
    for i, bigram in enumerate(unique_bigrams):
        if bigram in bigrams:
            input_vector[i] = 1

    predicted_indices = model.predict([input_vector])
    predictions = [term_index_map[idx] for idx in predicted_indices]

    return predictions[:5]
