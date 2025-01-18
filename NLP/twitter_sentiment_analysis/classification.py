import logging, sys
import re
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, hstack


"""
This classifier reads twitter sentences and performs sentimental analysis.
It identify if the sentence is positive or negative using Naive Bayes and Logistic Regression approaches.
It removes stop words, tokenize each word, and count the number of occurrences.
Models are built from the scratch instead of using any libraries.

This code referenced below sites.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
https://beckernick.github.io/logistic-regression-from-scratch/
"""

class Classifier(object):
    def __init__(self):
        self.train_data = "train.tsv"
        self.dev_data = "dev.tsv"
        self.test_data = "test.unlabeled.tsv"
        self.labels = ['0', '1']
        self.stop_words = []
        with open('stopwords.txt', 'r') as file:
            for line in file.readlines():
                self.stop_words.append(line.strip())

    def better_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"(@[a-zA-Z0-9_]+)|(\w+:\/\/\S+)", " ", text)
        text = re.sub('[^A-Za-z0-9 ]+', '', text)
        features = text.split()
        filtered_features = []
        for feature in features:
            if feature not in self.stop_words:
                filtered_features.append(feature)
        return filtered_features

    def tokenize(self, text):
        features = text.split()
        return features

    def parse_lines(self, file):
        instances = []
        with open(file) as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab', quoting=csv.QUOTE_NONE)
            for i in reader:
                instances.append(i)
        return instances

    def write_output(self, ids, predictions, model="bayes"):
        with open("test.labeled_{}.csv".format(model), 'w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['instance_id', 'class'])
            writer.writerows(zip(ids, predictions))


class BayesClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.likelihood = {}
        self.likelihood['0'] = {}
        self.likelihood['1'] = {}
        self.label_0_total = 0
        self.label_1_total = 0
            # logging.debug(self.stop_words)

    def reset(self):
        self.likelihood = {}
        self.likelihood['0'] = {}
        self.likelihood['1'] = {}
        self.label_0_total = 0
        self.label_1_total = 0

    def train(self, smoothing_alpha):
        # open file
        with open(self.train_data, 'r') as file:
            instances = file.readlines()
            for instance in instances[1:]:
                data = re.split(r'\t+', instance.strip())
                id = data[0]
                text = data[1]
                this_label = data[2]
                features = self.tokenize(text)
                # features = self.better_tokenize(text)
                #logging.debug(id)
                #logging.debug(features)
                for feature in features:
                    for label in self.labels:
                        if feature not in self.likelihood[label]:
                            if label == this_label:
                                self.likelihood[label][feature] = 1 + smoothing_alpha
                            else:
                                self.likelihood[label][feature] = smoothing_alpha
                        else:
                            if label == this_label:
                                self.likelihood[label][feature] += 1

                    if this_label == '0':
                        self.label_0_total += 1
                    else:
                        self.label_1_total += 1

        self.label_0_total += 1
        self.label_1_total += 1
        #logging.debug(self.likelihood)
        logging.debug("Done training Alpha: {}".format(smoothing_alpha))

    def classify(self, word_list):
        total = self.label_0_total + self.label_1_total

        posterior_1 = None
        posterior_0 = None
        for word in word_list:
            if (word in self.likelihood['0']) and (word in self.likelihood['1']):
                p_x = float((self.likelihood['0'][word] + self.likelihood['1'][word]) / total)
                p_1_x_c = self.likelihood['1'][word] / self.label_1_total
                p_1_c = self.label_1_total / total
                prob = float((p_1_x_c * p_1_c) / p_x)
                if posterior_1 is None:
                    posterior_1 = prob
                else:
                    posterior_1 = posterior_1 * prob
                # logging.debug("posterior1: {} prob: {} p_x: {} p_1_x_c: {} p_1_c: {}".format(posterior_1, prob, p_x, p_1_x_c, p_1_c))

                p_0_x_c = self.likelihood['0'][word] / self.label_0_total
                p_0_c = self.label_0_total / total
                prob = float((p_0_x_c * p_0_c) / p_x)
                if posterior_0 is None:
                    posterior_0 = prob
                else:
                    posterior_0 = posterior_0 * prob
                # logging.debug("posterior0: {} prob: {} p_x: {} p_0_x_c: {} p_0_c: {}".format(posterior_0, prob, p_x, p_0_x_c, p_0_c))

        #logging.debug("final posterior1: {}".format(posterior_1))
        #logging.debug("final posterior0: {}".format(posterior_0))

        if posterior_0 is None:
            posterior_0 = 0
        if posterior_1 is None:
            posterior_1 = 0

        if posterior_1 >= posterior_0:
            label = 1
        else:
            label = 0

        return label

    def predict_test(self):
        instances = self.parse_lines(self.test_data)
        ids = [d['instance_id'] for d in instances]
        docs = [d['text'] for d in instances]
        predictions = []
        for doc in docs:
            features = self.tokenize(doc)
            pred = self.classify(features)
            predictions.append(pred)

        # logging.debug("Test Prediction: {}".format(predictions))
        self.write_output(ids, predictions)

    def process(self, alp):
        y_pred = []
        y_true = []
        self.reset()
        self.train(alp)
        with open(self.dev_data, 'r') as file:
            instances = file.readlines()
            total = len(instances) - 1
            match = 0
            for instance in instances[1:]:
                data = re.split(r'\t+', instance.strip())
                id = data[0]
                text = data[1]
                actual = int(data[2])
                features = self.tokenize(text)
                # features = self.better_tokenize(text)
                label = self.classify(features)
                y_true.append(actual)
                y_pred.append(label)
                f1 = f1_score(y_true, y_pred)
                if actual == label:
                    match += 1
            result = float(match / total)
            logging.debug("Bayes Performance F1: {} ".format(f1))
            self.predict_test()
        return f1

class LogisticRegression(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.train_inst = self.parse_lines(self.train_data)
        self.dev_inst = self.parse_lines(self.dev_data)
        self.test_inst = self.parse_lines(self.test_data)
        self.vocab = {}

    def __reset(self):
        self.vocab = {}

    def __build_csr_matrix(self, instances, pred=False):
        docs = [d['text'] for d in instances]
        # logging.debug(doc)
        indptr = [0]
        indices = []
        data = []

        for d in docs:
            words = self.better_tokenize(d)
            # logging.debug(words)
            for word in words:
                if not pred:
                    index = self.vocab.setdefault(word, len(self.vocab))
                    indices.append(index)
                    data.append(1)
                else:
                    if word in self.vocab:
                        index = self.vocab.get(word)
                        indices.append(index)
                        data.append(1)

            indptr.append(len(indices))

        #logging.debug("Instances len: {} Vocab len: {}".format(len(instances), len(self.vocab)))
        matrix = csr_matrix((data, indices, indptr), shape=(len(instances), len(self.vocab)), dtype=int)

        return matrix

    def __sigmoid(self, scores):
        return (1 / (1+np.exp(-scores)))

    def __log_likelihood(self, features, target, weights):
        scores = features.dot(weights)
        ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
        return ll

    def __logistic_regression(self, features, target, learning_rate, num_steps):
        intercept = np.ones((features.shape[0], 1))
        features = hstack((intercept, features))
        # logging.debug(features)

        weights = np.zeros(features.shape[1])
        steps = []
        lls = []
        for step in range(num_steps):
            scores = features.dot(weights)
            #logging.debug("feature_0: {}".format(features[0]))
            #logging.debug("Weight: {}".format(weights))
            #logging.debug("score: {}".format(scores[0]))

            predictions = self.__sigmoid(scores)
            # logging.debug("pred: {}".format(predictions))

            #update gradient
            output_error = target - predictions
            gradient = features.T.dot(output_error)
            weights += learning_rate * gradient

            # print log-likelihood
            if step % 10000 == 0:
                ll = self.__log_likelihood(features, target, weights)
                logging.debug("Step: {} LL: {}".format(step, ll))
                steps.append(step)
                lls.append(ll)
            stats = {"rate": learning_rate, "steps": steps, "lls": lls}

        return weights, stats

    def __predict(self, weights):
        features = self.__build_csr_matrix(self.dev_inst, pred=True)
        actual = np.array([int(d['class']) for d in self.dev_inst])

        intercept = np.ones((features.shape[0], 1))
        features = hstack((intercept, features))
        scores = features.dot(weights)
        #logging.debug(features)
        predictions = self.__sigmoid(scores)
        #logging.debug(predictions)
        logging.debug("Dev Instance len: {} Pred len: {}".format(len(self.dev_inst), len(predictions)))
        pred_label =[]
        for pred in predictions:
            label = 1 if pred >= 0.5 else 0
            pred_label.append(label)
        f1 = f1_score(actual, pred_label)
        return f1

    def __predict_test(self, weights):
        features = self.__build_csr_matrix(self.test_inst, pred=True)
        intercept = np.ones((features.shape[0], 1))
        features = hstack((intercept, features))

        scores = features.dot(weights)
        pred_scores = self.__sigmoid(scores)
        logging.debug("Test Instances len: {} Pred len: {}".format(len(self.dev_inst), len(pred_scores)))
        ids = [d['instance_id'] for d in self.test_inst]
        predictions =[]
        for score in pred_scores:
            label = 1 if score >= 0.5 else 0
            predictions.append(label)
        self.write_output(ids, predictions, 'logistic')

    def process(self, num_steps=300000, learning_rate=5e-5):
        self.__reset()
        features = self.__build_csr_matrix(self.train_inst)
        target = np.array([int(d['class']) for d in self.train_inst])
        #logging.debug(features)
        #logging.debug("Target: {} ".format(target))
        weights, stats = self.__logistic_regression(features, target, learning_rate, num_steps)
        f1 = self.__predict(weights)
        logging.debug("Logistics Performance: {}".format(f1))
        self.__predict_test(weights)

        return stats


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    #alp_list = [0, 0.1, 0.3, 0.7, 1, 1.5, 2, 2.5, 3]
    alp_list = [0.1]
    per_list = []
    bayes = BayesClassifier()
    for alp in alp_list:
        f1 = bayes.process(alp)
        per_list.append(f1)
    #plt.plot(alp_list, per_list)
    #plt.show()

    # learning_rate = [0.005, 5e-5, 5e-10]
    learning_rate = [5e-5]
    results = []
    logistic = LogisticRegression()
    for rate in learning_rate:
        result = logistic.process(learning_rate=rate)
        results.append(result)

    #plt.plot(results[0]['steps'], results[0]['lls'], 'b')
    #plt.plot(results[1]['steps'], results[1]['lls'], 'r')
    #plt.plot(results[2]['steps'], results[2]['lls'], 'g')
    #plt.show()
