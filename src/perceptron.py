from typing import List, Dict, Set, DefaultDict, Callable, Tuple
from random import shuffle
from abc import abstractmethod
from util import read_train_data, sample_train_data_filename, train_data_filename, get_text_ngrams, \
    read_dev_data, read_dev_key_data, dev_data_filename, dev_data_key_filename, \
    get_performance_measure, prepare_train_data, dev_decode, get_cbow, get_sentiment_features, \
    get_authenticity_features, write_model, vanilla_model_filename, avg_model_filename, read_model, \
    write_output, output_filename, decode_auth_class, decode_sent_class, pprint
from collections import Counter, defaultdict
import time
import random
from statistics import mean
import json
import sys

ITERATIONS = 100
FREQ_LIMIT = 1
RARE_WORD_ID = "RARE_WORD_ID"
sentiment_model_name = "sentiment"
authentication_model_name = "authentication"


class Perceptron:
    """
    binary classifier, takes classes as +1, -1
    class encoding is not implicit
    """

    def __init__(self, cbow: Dict[str, int], bias: float, handle_rareword: bool):
        self.weights = self.initialize_weights(cbow)
        self.bias = bias
        self.cbow = cbow
        self.rares = self.get_rare_words(self.cbow)
        self.handle_rareword = handle_rareword

    def initialize_weights(self, cbow):
        if not cbow:
            return {}
        weights = {word: 0 for word in cbow}
        weights[RARE_WORD_ID] = 0
        return weights

    @staticmethod
    def get_rare_words(cbow: Dict[str, float]):
        rares = set()
        if cbow:
            for word, freq in cbow.items():
                if freq <= FREQ_LIMIT:
                    rares.add(word)
        return rares

    def activation(self, regress_value):
        if regress_value > 0:
            return 1
        return -1

    def compute(self, features: Dict[str, float]):
        regress_value = 0
        for (word, count) in features.items():
            if word in self.weights:
                regress_value += self.weights[word] * count

            if self.handle_rareword and word not in self.weights:
                regress_value += self.weights[RARE_WORD_ID]*count

            if self.handle_rareword and word in self.rares:
                regress_value += self.weights[RARE_WORD_ID]*count

        return self.activation(regress_value + self.bias)

    @abstractmethod
    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str], Dict[str, float]]) -> int:
        pass

    def train(self, data: List[Tuple[int, str]], max_iterations: int, get_feature: Callable[[str],
                Dict[str, float]]):
        for iter in range(max_iterations):
            wrongs = self.train_single_iteration(data, get_feature)
            if wrongs == 0:
                break

    def predict(self, data: str, get_feature) -> int:
        return self.compute(get_feature(data))

    def update_parameters(self, features: Dict[str, float], cls: int):
        for (word, count) in features.items():
            self.weights[word] += cls * count
            if self.handle_rareword and word in self.rares:
                self.weights[RARE_WORD_ID] += cls*count
        self.bias += cls

    def store_model(self):
        return {"weights": self.weights,
                "bias": self.bias,
                "rares": list(self.rares),
                "handle_rareword": self.handle_rareword}

    @staticmethod
    def build_from_file(filename):
        with open(filename, 'r') as fp:
            return Perceptron.builder(json.load(fp))

    @staticmethod
    def builder(model_data):
        p = Perceptron(None, None, None)
        p.weights = model_data["weights"]
        p.bias = model_data["bias"]
        p.rares = set(model_data["rares"])
        p.handle_rareword = model_data["handle_rareword"]
        return p




class Vanilla_Perceptron(Perceptron):
    def __init__(self, cbow: Dict[str, int], bias: float, handle_rareword: bool):
        super().__init__(cbow, bias, handle_rareword)

    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str],
        Dict[str, float]]) -> int:
        wrong_counts = 0
        for (cls, text) in data:
            features = get_feature(text)
            prediction = self.compute(features)
            if prediction * cls <= 0:
                self.update_parameters(features, cls)
                wrong_counts += 1
        return wrong_counts


class Avg_Perceptron(Perceptron):

    def __init__(self, cbow: Dict[str, int], bias: float, handle_rareword: bool):
        super(Avg_Perceptron, self).__init__(cbow, bias, handle_rareword)
        self.cached_weights = dict(self.weights)
        self.cached_bias = bias
        self.count = 0

    def update_cache(self, features, cls):
        for word, weight in features.items():
            self.cached_weights[word] += cls * weight * self.count
            if self.handle_rareword and word in self.rares:
                self.cached_weights[RARE_WORD_ID] += cls*weight*self.count
        self.cached_bias += cls*self.count

    def avg_weights(self):
        for word, weight in self.weights.items():
            self.weights[word] -= self.cached_weights[word]/self.count
        self.bias -= self.cached_bias/self.count

    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str],
                        Dict[str, float]]) -> int:
        wrong_count = 0
        for (cls, text) in data:
            features = get_feature(text)
            prediction = self.compute(features)
            self.count+=1
            if prediction * cls <= 0:
                wrong_count += 1
                self.update_parameters(features, cls)
                self.update_cache(features, cls)
        return wrong_count

    def train(self, data: List[Tuple[int, str]], max_iterations: int, get_feature: Callable[[str],
                Dict[str, float]]):
        super().train(data, max_iterations, get_feature)
        self.avg_weights()

def model_build(sentiment_data, authenticity_data, text):
    sent_bag_of_words = get_cbow(text, False)
    auth_bag_of_words = get_cbow(text, True)

    vp_sent = Vanilla_Perceptron(sent_bag_of_words, 0, True)
    vp_sent.train(sentiment_data, ITERATIONS, get_sentiment_features)

    vp_auth = Vanilla_Perceptron(auth_bag_of_words, 0, False)
    vp_auth.train(authenticity_data, ITERATIONS, get_authenticity_features)


    ap_sent = Avg_Perceptron(sent_bag_of_words, 0, True)
    ap_sent.train(sentiment_data, ITERATIONS, get_sentiment_features)

    ap_auth = Avg_Perceptron(auth_bag_of_words, 0, False)
    ap_auth.train(authenticity_data, ITERATIONS, get_sentiment_features)

    return vp_sent, vp_auth, ap_sent, ap_auth



def model_train(filename: str):
    sentiment_data, authenticity_data, text = prepare_train_data(filename)
    vp_sent, vp_auth, ap_sent, ap_auth = model_build(sentiment_data, authenticity_data, text)
    vanilla_models = {sentiment_model_name: vp_sent.store_model(), authentication_model_name: vp_auth.store_model()}
    write_model(vanilla_models, vanilla_model_filename)
    average_model = {sentiment_model_name: ap_sent.store_model(), authentication_model_name: ap_auth.store_model()}
    write_model(average_model, avg_model_filename)


def build_model(model_data):
    sent_model_data = model_data[sentiment_model_name]
    auth_model_data = model_data[authentication_model_name]
    return Perceptron.builder(sent_model_data), Perceptron.builder(auth_model_data)


def model_predict_data(sent_model, auth_model, data):
    predictions = []
    # print(data)
    for (reviewId, line) in data:
        sent_cls = sent_model.predict(line, get_sentiment_features)
        auth_cls = auth_model.predict(line, get_authenticity_features)
        predictions.append(
            " ".join([reviewId, decode_auth_class(auth_cls), decode_sent_class(sent_cls)]))
    return predictions

def model_predict(model_filename: str, data_filename: str):
    data = read_dev_data(data_filename)
    models = read_model(model_filename)
    sent_model, auth_model = build_model(models)
    predictions = model_predict_data(sent_model, auth_model, data)
    write_output(predictions)

def train_and_predict(p: Perceptron, train_data: List[Tuple[int, str]],
                      test_data: List[Tuple[str, str]], get_feature: Callable[[str], Dict[str, float]]):
    p.train(train_data, ITERATIONS, get_feature)
    return {reviewId: p.predict(text, get_feature) for (reviewId, text) in test_data}

def vanilla_models(sent_train_data: List[Tuple[int, str]], auth_train_data: List[Tuple[int, str]],
                   test_data: List[Tuple[str, str]], text: List[str]):


    sent_bag_of_words = get_cbow(text, False)
    auth_bag_of_words = get_cbow(text, True)


    vp_sent = Vanilla_Perceptron(sent_bag_of_words, 0, True)
    vp_auth = Vanilla_Perceptron(auth_bag_of_words, 0, False)

    sent_predictions = train_and_predict(vp_sent, sent_train_data, test_data, get_sentiment_features)
    auth_predictions = train_and_predict(vp_auth, auth_train_data, test_data, get_authenticity_features)
    return sent_predictions, auth_predictions

def avg_models(sent_train_data: List[Tuple[int, str]], auth_train_data: List[Tuple[int, str]],
               test_data: List[Tuple[str, str]], text: List[str]):

    sent_bag_of_words = get_cbow(text, False)
    auth_bag_of_words = get_cbow(text, True)


    ap_sent = Avg_Perceptron(sent_bag_of_words, 0, True)
    ap_auth = Avg_Perceptron(auth_bag_of_words, 0, False)

    sent_predictions = train_and_predict(ap_sent, sent_train_data, test_data, get_sentiment_features)
    auth_predictions = train_and_predict(ap_auth, auth_train_data, test_data, get_authenticity_features)

    return sent_predictions, auth_predictions




def pre_processing():
    sentiment_data, authenticity_data, text = prepare_train_data(train_data_filename)
    dev_data = read_dev_data(dev_data_filename)
    dev_key = read_dev_key_data(dev_data_key_filename)
    sent_bag_of_words = get_cbow(text, False)
    auth_bag_of_words = get_cbow(text, True)

    # vanilla model
    vsent_predictions, vauth_predictions = vanilla_models(sentiment_data, authenticity_data, dev_data, text)

    vsent_predictions, sent_gold = dev_decode(vsent_predictions, dev_key, 1)
    vauth_predictions, auth_gold = dev_decode(vauth_predictions, dev_key, 0)

    _, _, vsent_pos_f1 = get_performance_measure(vsent_predictions, sent_gold, 1)
    _, _, vsent_neg_f1 = get_performance_measure(vsent_predictions, sent_gold, -1)
    _, _, vauth_pos_f1 = get_performance_measure(vauth_predictions, auth_gold, 1)
    _, _, vauth_neg_f1 = get_performance_measure(vauth_predictions, auth_gold, -1)

    vanilla_model_preformance = mean([vsent_pos_f1, vsent_neg_f1, vauth_pos_f1, vauth_neg_f1])

    asent_predictions, aauth_predictions = avg_models(sentiment_data, authenticity_data, dev_data, text)

    asent_predictions, sent_gold = dev_decode(asent_predictions, dev_key, 1)
    aauth_predictions, auth_gold = dev_decode(aauth_predictions, dev_key, 0)

    _, _, asent_pos_f1 = get_performance_measure(asent_predictions, sent_gold, 1)
    _, _, asent_neg_f1 = get_performance_measure(asent_predictions, sent_gold, -1)
    _, _, aauth_pos_f1 = get_performance_measure(aauth_predictions, auth_gold, 1)
    _, _, aauth_neg_f1 = get_performance_measure(aauth_predictions, auth_gold, -1)

    avg_model_performance = mean([asent_pos_f1, asent_neg_f1, aauth_pos_f1, aauth_neg_f1])

    return vanilla_model_preformance, avg_model_performance
    # return 0,0

def trail_run():
    vanilla_scores = []
    avg_scores = []
    for i in range(0, 20):
        sf1, af1 = pre_processing()
        print(sf1, af1)
        vanilla_scores.append(sf1)
        avg_scores.append(af1)
    print("average scores")
    print(mean(vanilla_scores))
    print(mean(avg_scores))

def learn_main():
    input_filename = sys.argv[1]
    model_train(input_filename)


def predict_main():
    model_filename = sys.argv[1]
    data_filename = sys.argv[2]
    model_predict(model_filename, data_filename)


if __name__ == '__main__':
    trail_run()