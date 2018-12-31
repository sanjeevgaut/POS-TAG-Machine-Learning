import nltk
import pdb
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import brown
from pickle import dump
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def features(sentence, index):
    currWord = sentence[index][0]
    if (index > 0):
        prevWord = sentence[index - 1][0]
    else:
        prevWord = '<START>'
    if (index < len(sentence)-1):
        nextWord = sentence[index + 1][0]
    else:
        nextWord = '<END>'
    return {
        'word': currWord,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'curr_is_title': currWord.istitle(),
        'prev_is_title': prevWord.istitle(),
        'next_is_title': nextWord.istitle(),
        'curr_is_lower': currWord.islower(),
        'prev_is_lower': prevWord.islower(),
        'next_is_lower': nextWord.islower(),
        'curr_is_upper': currWord.isupper(),
        'prev_is_upper': prevWord.isupper(),
        'next_is_upper': nextWord.isupper(),
        'curr_is_digit': currWord.isdigit(),
        'prev_is_digit': prevWord.isdigit(),
        'next_is_digit': nextWord.isdigit(),
        'curr_prefix-1': currWord[0],
        'curr_prefix-2': currWord[:2],
        'curr_prefix-3': currWord[:3],
        'curr_suffix-1': currWord[-1],
        'curr_suffix-2': currWord[-2:],
        'curr_suffix-3': currWord[-3:],
        'prev_prefix-1': prevWord[0],
        'prev_prefix-2': prevWord[:2],
        'prev_prefix-3': prevWord[:3],
        'prev_suffix-1': prevWord[-1],
        'prev_suffix-2': prevWord[-2:],
        'prev_suffix-3': prevWord[-3:],
        'next_prefix-1': nextWord[0],
        'next_prefix-2': nextWord[:2],
        'next_prefix-3': nextWord[:3],
        'next_suffix-1': nextWord[-1],
        'next_suffix-2': nextWord[-2:],
        'next_suffix-3': nextWord[-3:],
        'prev_word': prevWord,
        'next_word': nextWord,
    }

def transformDataset(sentences):
    wordFeatures = []
    wordLabels = []
    for sent in sentences:
        for index in range(len(sent)):
            wordFeatures.append(features(sent, index))
            wordLabels.append(sent[index][1])
            #pdb.set_trace()
            #print(wordFeatures,wordLabels)
        return wordFeatures, wordLabels

def trainDecisionTree(trainFeatures, trainLabels):
    
    clf = make_pipeline(DictVectorizer(sparse=False), DecisionTreeClassifier(criterion='entropy'))
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean()

def trainNaiveBayes(trainFeatures, trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=False), MultinomialNB())
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean()

def trainNN(trainFeatures, trainLabels):
    clf = make_pipeline(DictVectorizer(sparse=False),
                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1))
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=5)
    clf.fit(trainFeatures, trainLabels)
    return clf, scores.mean()


def ngramTagger(train_sents, n=0, defaultTag='NN'):
    
    t0 = nltk.DefaultTagger(defaultTag)
    if (n <= 0):
        return t0
    elif (n == 1):
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        return t1
    elif (n == 2):
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        return t2
    else:
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        t3 = nltk.TrigramTagger(train_sents, backoff=t2)
        return t3


brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.7)

tags = [tag for (word, tag) in brown.tagged_words()]

defaultTag = nltk.FreqDist(tags).max()
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

tagger = ngramTagger(train_sents, 2, defaultTag)
print(tagger.evaluate(test_sents))

trainFeatures, trainLabels = transformDataset(train_sents)
testFeatures, testLabels = transformDataset(test_sents)

tree_model, tree_model_cv_score = trainDecisionTree(trainFeatures[:30000], trainLabels[:30000])

print(tree_model_cv_score)
print(tree_model.score(testFeatures, testLabels))
nb_model, nb_model_cv_score = trainNaiveBayes(trainFeatures[:30000], trainLabels[:30000])
print(nb_model_cv_score)
print(nb_model.score(testFeatures, testLabels) )
nn_model, nn_model_cv_score = trainNN(trainFeatures[:30000], trainLabels[:30000])
print(nn_model_cv_score)
print(nn_model.score(testFeatures, testLabels))
