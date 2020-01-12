from __future__ import division
import os
import numpy as np
import string
import math
from sklearn import svm

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

    #data_dir = "/usr/groups/mphil/L90/data-tagged/"
    data_dir = "/Users/cindyliu/Documents/Cambridge/L90/Code/data-tagged/"

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

#choose uni-grams, bi-grams, or both
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

#non-smoothed
V, prior, condprob = trainNB(train_data, train_labels, 0)
predictions = list()
for doc in test_data:
    predictions.append(predictNB(doc, V, prior, condprob, ngrams))
correct = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        correct += 1
accuracy = correct / len(predictions)
print("Prediction accuracy (NB no smoothing, no cross-validation): " + str(accuracy))

#smoothed
V, prior, condprob = trainNB(train_data, train_labels, 1)
predictions = list()
for doc in test_data:
    predictions.append(predictNB(doc, V, prior, condprob, ngrams))
correct = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        correct += 1
accuracy = correct / len(predictions)
print("Prediction accuracy (NB K = 1, no cross-validation): " + str(accuracy))
print(predictions)

#SVM with BOW
clf = svm.SVC(gamma='scale', kernel = 'rbf', C = 1)
V_order = dict()
i = 0
for w in V:
    V_order[w] = i
    i += 1
svm_train_data = []
for doc in train_data:
    doc_vector = np.zeros(len(V))
    for w in doc:
        if w in V_order.keys():
            doc_vector[V_order[w]] = 1
    svm_train_data.append(doc_vector)
svm_test_data = []
for doc in test_data:
    doc_vector = np.zeros(len(V))
    for w in doc:
        if w in V_order.keys():
            doc_vector[V_order[w]] = 1
    svm_test_data.append(doc_vector)
clf.fit(svm_train_data, train_labels)
score = clf.score(svm_test_data, test_labels)
print("SVM score: " + str(score))
print(list(clf.predict(svm_test_data)))


#with 10-fold CV
split_data, split_labels = train_test_split_CV(data, labels)
accuracies_ns = []
accuracies_s = []
accuracies_svm = []
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

    #non-smoothed
    V, prior, condprob = trainNB(train_data, train_labels, 0)
    for doc in test_data:
        predictions_ns.append(predictNB(doc, V, prior, condprob, ngrams))
    correct = 0
    for i in range(len(predictions_ns)):
        if predictions_ns[i] == test_labels[i]:
            correct += 1
    accuracies_ns.append(correct / len(predictions_ns))

    #smoothed
    V, prior, condprob = trainNB(train_data, train_labels, 1)
    for doc in test_data:
        predictions_s.append(predictNB(doc, V, prior, condprob, ngrams))
    correct = 0
    for i in range(len(predictions_s)):
        if predictions_s[i] == test_labels[i]:
            correct += 1
    accuracies_s.append(correct / len(predictions_s))

    #SVM
    clf = svm.SVC(gamma='scale', kernel='rbf', C=1)
    V_order = dict()
    i = 0
    for w in V:
        V_order[w] = i
        i += 1
    svm_train_data = []
    for doc in train_data:
        doc_vector = np.zeros(len(V))
        for w in doc:
            if w in V_order.keys():
                doc_vector[V_order[w]] = 1
        svm_train_data.append(doc_vector)
    svm_test_data = []
    for doc in test_data:
        doc_vector = np.zeros(len(V))
        for w in doc:
            if w in V_order.keys():
                doc_vector[V_order[w]] = 1
        svm_test_data.append(doc_vector)
    clf.fit(svm_train_data, train_labels)
    score = clf.score(svm_test_data, test_labels)
    accuracies_svm.append(score)

print("Non-smoothed: ")
print("All accuracies: " + str(accuracies_ns))
print("Average prediction accuracy: " + str(sum(accuracies_ns) / 10) + ", var: " + str(np.var(accuracies_ns)))

print("-------")

print("Smoothed: ")
print("All accuracies: " + str(accuracies_s))
print("Average prediction accuracy: " + str(sum(accuracies_s) / 10) + ", var: " + str(np.var(accuracies_s)))

print("-------")

print("SVM: ")
print("All accuracies: " + str(accuracies_svm))
print("Average prediction accuracy: " + str(sum(accuracies_svm) / 10) + ", var: " + str(np.var(accuracies_svm)))