from typing import List, Dict, Set, DefaultDict, Callable, NoReturn, Tuple
from random import shuffle
from abc import abstractmethod
from util import read_train_data, sample_train_data_filename, train_data_filename, get_unigrams, \
    read_dev_data, read_dev_key_data, dev_data_filename, dev_data_key_filename, \
    get_performance_measure, prepare_train_data, dev_decode, get_cbow
from collections import Counter
import time
import random

ITERATIONS = 25


class Perceptron:
    """
    binary classifier, takes classes as +1, -1
    class encoding is not implicit
    """

    def __init__(self, weights: Dict[str, float], bias: float):
        self.weights = weights
        self.bias = bias
        self.wl = len(weights)

    def activation(self, regress_value):
        if regress_value > 0:
            return 1
        return -1

    def compute(self, features: Dict[str, float]):
        regress_value = 0
        for (word, count) in features.items():
            if word in self.weights:
                regress_value += self.weights[word] * count
            else:
                print("unknown words {}".format(word))
                "ToDo: implement unknown words handling"
        return self.activation(regress_value + self.bias)

    @abstractmethod
    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str], Dict[str, float]]) -> NoReturn:
        pass

    def train(self, data: List[Tuple[int, str]], max_iterations: int, get_feature: Callable[[str],
                Dict[str, float]]) -> NoReturn:
        start = time.time()
        for iter in range(max_iterations):
            shuffle(data)
            self.train_single_iteration(data, get_feature)
        print("took {} seconds".format((time.time() - start)))

    def predict(self, data: str, get_feature) -> int:
        return self.compute(get_feature(data))

    def update_parameters(self, features: Dict[str, float], cls: int):
        for (word, count) in features.items():
            self.weights[word] += cls * count
        self.bias += cls


class Vanilla_Perceptron(Perceptron):
    def __init__(self, weights: Dict[str, float], bias: float):
        super().__init__(weights, bias)

    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str], Dict[str, float]]) -> NoReturn:
        wrong_counts = 0
        for (cls, text) in data:
            features = get_feature(text)
            prediction = self.compute(features)
            if prediction * cls <= 0:
                self.update_parameters(features, cls)
                wrong_counts += 1
        print("wrong predictions {}".format(wrong_counts))


class Avg_Perceptron(Perceptron):

    def __init__(self, weights: Dict[str, float], bias: float):
        super(Avg_Perceptron, self).__init__(weights, bias)
        self.cached_weights = dict(weights)
        self.cached_bias = bias
        self.count = 1

    def update_cache(self, features, cls):
        for word, count in features.items():
            self.cached_weights[word] += cls * count
        self.cached_bias += cls

    def avg_weights(self):
        for word, weight in self.weights.items():
            self.weights[word] -= self.cached_weights[word]/self.count
        self.bias -= self.cached_bias/self.count

    def train_single_iteration(self, data: List[Tuple[int, str]], get_feature: Callable[[str],
                        Dict[str, float]]) -> NoReturn:
        for (cls, text) in data:
            features = get_feature(text)
            prediction = self.compute(features)
            self.count += 1
            if prediction * cls <= 0:
                self.update_parameters(features, cls)
                self.update_cache(features, cls)

    def train(self, data: List[Tuple[int, str]], max_iterations: int, get_feature: Callable[[str],
                Dict[str, float]]):
        super().train(data, max_iterations, get_feature)
        self.avg_weights()


def get_feature(text):
    result = Counter(get_unigrams(text))
    return result


def train_and_predict(p: Perceptron, train_data: List[Tuple[int, str]], test_data: List[Tuple[str, str]]):
    p.train(train_data, ITERATIONS, get_feature)
    return {reviewId: p.predict(text, get_feature) for (reviewId, text) in test_data}

def vanilla_models(sent_train_data: List[Tuple[int, str]], auth_train_data: List[Tuple[int, str]],
                   test_data: List[Tuple[str, str]], bag_of_words: Dict[str, float]):
    vp_sent = Vanilla_Perceptron({word: 0 for word in bag_of_words}, 0)
    # vp_auth = Vanilla_Perceptron({word: 0 for word in bag_of_words}, 0)

    sent_predictions = train_and_predict(vp_sent, sent_train_data, test_data)
    # auth_predictions = train_and_predict(vp_auth, auth_train_data, test_data)
    return sent_predictions, []

def avg_models(sent_train_data: List[Tuple[int, str]], auth_train_data: List[Tuple[int, str]],
               test_data: List[Tuple[str, str]], bag_or_words: Dict[str, float]):

    ap_sent = Avg_Perceptron({word: 0 for word in bag_or_words}, 0)
    # ap_auth = Avg_Perceptron({word: 0 for word in bag_or_words}, 0)

    sent_predictions = train_and_predict(ap_sent, sent_train_data, test_data)
    # auth_predictions = train_and_predict(ap_auth, auth_train_data, test_data)
    return sent_predictions, []



def pre_processing():
    sentiment_data, authenticity_data, text = prepare_train_data()
    dev_data = read_dev_data(dev_data_filename)
    dev_key = read_dev_key_data(dev_data_key_filename)
    bag_of_words = get_cbow(text)


    # vanilla model
    sent_predictions, auth_predictions = vanilla_models(sentiment_data, authenticity_data, dev_data, bag_of_words)

    # average model
    # sent_predictions, auth_predictions = avg_models(sentiment_data, authenticity_data, dev_data, bag_of_words)

    sent_predictions, sent_gold = dev_decode(sent_predictions, dev_key, 1)
    # auth_predictions, auth_gold = dev_decode(auth_predictions, dev_key, 0)

    _, _, sent_pos_f1 = get_performance_measure(sent_predictions, sent_gold, 1)
    _, _, sent_neg_f1 = get_performance_measure(sent_predictions, sent_gold, -1)
    # _, _, auth_pos_f1 = get_performance_measure(auth_predictions, auth_gold, 1)
    # _, _, auth_neg_f1 = get_performance_measure(auth_predictions, auth_gold, -1)

    # performance check
    print(get_performance_measure(sent_predictions, sent_gold, 1))
    print(get_performance_measure(sent_predictions, sent_gold, -1))
    # print(get_performance_measure(auth_predictions, auth_gold, 1))
    # print(get_performance_measure(auth_predictions, auth_gold, -1))
    # print((sent_pos_f1 + sent_neg_f1 + auth_pos_f1 + auth_neg_f1)/4)


# for i in range(0, 10):
pre_processing()


