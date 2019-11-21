from __future__ import division
import os
import numpy as np
import string
import math

def extractVocab(docs):
    """
    Converts the contents of the documents into a vocabulary (as a set).
    """
    V = set()
    for d in docs:
        for word in d:
            V.add(word)
    return V


def concatenateDocs(docs, labels):
    """
    Concatenates all docs with the same label.
    """
    text = [[], []]
    for d, c in zip(docs, labels):
        text[c] += d
    return text


def countWords(text, vocab):
    """
    Creates a dictionary with all words and their respective counts in the
    given text.
    """
    counts = [{k:0 for k in vocab}, {k:0 for k in vocab}]
    for c in [0, 1]:
        for t in text[c]:
            counts[c][t] += 1
    return counts


def trainNB(train_data, train_labels, K = 0):
    """
    Trains the Naive Bayes classifier; includes Laplace smoothing.
    :param K: Laplace smoothing constant (default is 0, without smoothing)
    :return: the vocabulary V, priors, and conditional probabilities
    """
    orig_V = extractVocab(train_data)
    N = len(train_data)
    prior = [0, 0]
    text = concatenateDocs(train_data, train_labels)
    counts = countWords(text, orig_V)
    #remove rare words (less than 4 instances) from vocab
    V = set()
    for t in orig_V:
        if len(t.split(" ")) == 1:
            if counts[0][t] + counts[1][t] >= 4:
                V.add(t)
        elif len(t.split(" ")) == 2:
            if counts[0][t] + counts[1][t] >= 7:
                V.add(t)
    condprob = [{k:0 for k in V}, {k:0 for k in V}]
    #NB training
    for c in [0, 1]:
        if c == 0:
            N_c = N - sum(train_labels)
        else:
            N_c = sum(train_labels)
        prior[c] = N_c / N
        total_count = 0
        for t in V:
            total_count += counts[c][t]
        for t in V:
            condprob[c][t] = (counts[c][t] + K) / (total_count + (K * len(V)))
    return V, prior, condprob


def predictNB(text, V, prior, condprob, ngrams = 0):
    """
    Makes prediction using the trained Naive Bayes classifier.
    """
    score = [0, 0]
    for c in [0, 1]:
        score[c] = np.log(prior[c])
        #unigrams
        if ngrams == 0 or ngrams == 1:
            for t in text:
                if t in V:
                    score[c] += np.log(condprob[c][t])
        #bigrams
        if ngrams == 0 or ngrams == 2:
            for t in text:
                if t in V:
                    score[c] += np.log(condprob[c][t])


    return score.index(max(score))

def read_in_data():
    """
    Reads in the review data from tokenized review files.
    :return: data (as a list of words for each document) and 
    labels (POS = 1, NEG = 0)
    """
    all_data_unigrams = []
    all_labels = []

    data_dir = "/usr/groups/mphil/L90/data-tagged/"
    #data_dir = "/Users/cindyliu/Documents/Cambridge/L90/Code/data-tagged/"

    #read-in individual words (unigrams)
    for C in ["POS", "NEG"]:
        dir = data_dir + C
        for file in os.listdir(dir):
            f = open(dir + '/' + file)
            #store each doc as list of words
            doc = []
            for line in f:
                #lower-case, take out punctuation
                word = line.split('\t')[0].lower() #tokenized data is tabbed
                if word[0] not in string.punctuation and word != "\n":
                    doc.append(word)
            all_data_unigrams.append(doc)
            #POS = 1, NEG = 0
            if C == "POS":
                all_labels.append(1)
            else:
                all_labels.append(0)

    #add bigrams
    all_data_bigrams = []
    for doc in all_data_unigrams:
        bigrams_doc = []
        for word1, word2 in zip(doc, doc[1:]):
            bigrams_doc.append(word1 + " " + word2)
        all_data_bigrams.append(bigrams_doc)

    #combine unigrams and bigrams
    all_data_combined = []
    for i in range(len(all_data_bigrams)):
        all_data_combined.append(all_data_unigrams[i] + all_data_bigrams[i])

    return all_data_unigrams, all_data_bigrams, all_data_combined, all_labels


def train_test_split(data, labels):
    """
    Splits data into training and test sets (without cross-validation)
    :return: training and test sets and labels
    """
    train_data = data[:900] + data[1000:1900]
    train_labels = labels[:900] + labels[1000:1900]
    test_data = data[900:1000] + data[1900:]
    test_labels = labels[900:1000] + labels[1900:]
    return train_data, train_labels, test_data, test_labels


def train_test_split_CV(data, labels):
    """
    Splits data into training and test sets using 10-fold round-robin CV
    :return: training and test sets and labels
    """
    split_data = [[] for i in range(10)]
    split_labels = [[] for i in range(10)]
    for i in range(10):
        for j in range(int(len(data) / 10)):
            split_data[i].append(data[j*10 + i])
            split_labels[i].append(labels[j*10 + i])
    return split_data, split_labels


data_unigrams, data_bigrams, data_combined, labels = read_in_data()

ngrams = 0
if ngrams == 1:
    data = data_unigrams
    print("uni-grams only")
elif ngrams == 2:
    data = data_bigrams
    print("bi-grams only")
else:
    data = data_combined
    print("both uni-grams and bi-grams")

#without CV
train_data, train_labels, test_data, test_labels = train_test_split(data, labels)

V, prior, condprob = trainNB(train_data, train_labels, 0)
predictions = list()
for doc in test_data:
    predictions.append(predictNB(doc, V, prior, condprob, ngrams))
correct = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        correct += 1
accuracy = correct / len(predictions)
print("Prediction accuracy (no smoothing, no cross-validation): " + str(accuracy))

V, prior, condprob = trainNB(train_data, train_labels, 1)
predictions = list()
for doc in test_data:
    predictions.append(predictNB(doc, V, prior, condprob, ngrams))
correct = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        correct += 1
accuracy = correct / len(predictions)
print("Prediction accuracy (K = 1, no cross-validation): " + str(accuracy))


#with 10-fold CV
split_data, split_labels = train_test_split_CV(data, labels)
accuracies_ns = []
accuracies_s = []
for i in range(10):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for j in range(10):
        if j != i:
            train_data += split_data[j]
            train_labels += split_labels[j]
    test_data = split_data[i]
    test_labels = split_labels[i]

    predictions_ns = list()
    predictions_s = list()

    V, prior, condprob = trainNB(train_data, train_labels, 0)
    for doc in test_data:
        predictions_ns.append(predictNB(doc, V, prior, condprob, ngrams))
    correct = 0
    for i in range(len(predictions_ns)):
        if predictions_ns[i] == test_labels[i]:
            correct += 1
    accuracies_ns.append(correct / len(predictions_ns))

    V, prior, condprob = trainNB(train_data, train_labels, 1)
    for doc in test_data:
        predictions_s.append(predictNB(doc, V, prior, condprob, ngrams))
    correct = 0
    for i in range(len(predictions_s)):
        if predictions_s[i] == test_labels[i]:
            correct += 1
    accuracies_s.append(correct / len(predictions_s))

print("Non-smoothed: ")
print("All accuracies: " + str(accuracies_ns))
print("Average prediction accuracy: " + str(sum(accuracies_ns) / 10) + ", var: " + str(np.var(accuracies_ns)))

print("-------")

print("Smoothed: ")
print("All accuracies: " + str(accuracies_s))
print("Average prediction accuracy: " + str(sum(accuracies_s) / 10) + ", var: " + str(np.var(accuracies_s)))

#sign test calculations
plus = 0
minus = 0
null = 0
q = 0.5

for i in range(len(accuracies_ns)):
    if accuracies_ns[i] > accuracies_s[i]:
        plus = plus + 1
    elif accuracies_ns[i] < accuracies_s[i]:
        minus = minus + 1
    else:
        null = null + 1

N = 2 * math.ceil(null / 2) + plus + minus
k = math.ceil(null / 2) + min(plus, minus)
p = 0
for i in range(int(k)):
    p = p + 2 * (math.factorial(N) / (math.factorial(i) * math.factorial(N - i))) * (q ** i) * ((1 - q) ** (N- i))
print("p-value: " + str(p))
