from typing import List, Dict, Set, DefaultDict, Callable, Tuple
from random import shuffle
from abc import abstractmethod
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

from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
import string
import os, sys, json

# from perceptron import Encoder

train_data_filename = "../data/train-labeled.txt"
sample_train_data_filename = "../data/sample-train-labeled.txt"
dev_data_filename = "../data/dev-text.txt"
dev_data_key_filename = "../data/dev-key.txt"
full_data_filename = "../data/full_data.txt"



vanilla_model_filename = "vanillamodel.txt"
avg_model_filename = "averagedmodel.txt"



output_filename = "percepoutput.txt"


punctuations = set(string.punctuation)

stop_words = {'i', 'or', 'besides', 'six', 'whom', 'either', 'being', 'when', 'always', 'even',
              'amongst', 'on', 'all', 'over', 'eight', 'back', 'has', 'have', 'less', 'ourselves',
              'a', 'about', 'my', 'seems', 'until', 'keep', 'toward', 'anyway', 'to', 'around',
              'beforehand', 'cannot', 'could', 'does', 'had', 'somehow', 'thus', 'am', 'if',
              'their', 'front', 'who', 'once', 'put', 'is', 'some', 'under', 'whole', 'well',
              'beyond', 'often', 'onto', 'see', 'sometimes', 'by', 'forty', 'fifteen', 'part',
              'everyone', 'than', 'these', 'can', 'twelve', 'another', 'been', 'next', 'same',
              'seeming', 'further', 'used', 'might', 'become', 'himself', 'our', 'twenty', 'ours',
              'us', 'thereafter', 'else', 'few', 'after', 'therefore', 'was', 'various', 'move',
              'several', 'elsewhere', 'would', 'among', 'so', 'nor', 'this', 'becoming', 'yourself',
              'each', 'also', 'mine', 'everything', 'for', 'done', 'empty', 'per', 'whereafter',
              'please', 'together', 'then', 'unless', 'full', 'however', 'give', 'no', 'below',
              'since', 'whereby', 'already', 'that', 'must', 'between', 'seemed', 'hereupon',
              'because', 'down', 'every', 'made', 'as', 'thru', 'neither', 'least', 'wherein',
              'both', 'here', 'and', 'indeed', 'therein', 'bottom', 'throughout', 'yourselves',
              'regarding', 'ca', 'ever', 'call', 'somewhere', 'there', 'whoever', 'whence', 'in',
              'serious', 'such', 'latterly', 'last', 'only', 'top', 'against', 'out', 'name',
              'much', 'along', 'herein', 'from', 'hers', 'two', 'into', 'while', 'without',
              'whether', 'became', 'anyhow', 'where', 'within', 'enough', 'hereby', 'four', 'very',
              'whatever', 'myself', 'again', 'alone', 'yours', 'should', 'them', 'nobody', 'nine',
              'nevertheless', 'three', 'up', 'moreover', 'why', 'afterwards', 'not', 'his',
              'sometime', 'first', 'never', 'go', 'otherwise', 'third', 'via', 'will', 'herself',
              'at', 'becomes', 'before', 'him', 'themselves', 'amount', 'your', 'did', 'are',
              'what', 'more', 'namely', 'perhaps', 'whenever', 'do', 'hereafter', 'just',
              'thereupon', 'too', 'anywhere', 'you', 'be', 'sixty', 'most', 'behind', 'mostly',
              'other', 'something', 'during', 'meanwhile', 'seem', 'though', 'although', 'latter',
              'get', 'anyone', 'itself', 'they', 'of', 'take', 'show', 'whither', 'none', 'yet',
              'she', 'wherever', 'ten', 'upon', 'beside', 'an', 'any', 'but', 'make', 'hence',
              'off', 'one', 'own', 'rather', 'someone', 'using', 'it', 'anything', 'may', 'others',
              'nothing', 'really', 'we', 'due', 'me', 'whose', 'everywhere', 're', 'former',
              'fifty', 'above', 'say', 'the', 'doing', 'still', 'thence', 'eleven', 'five', 'her',
              'quite', 'thereby', 'whereupon', 'many', 'almost', 'except', 'hundred', 'nowhere',
              'whereas', 'none', 'with', 'across', 'which', 'those', 'towards', 'how', 'side',
              'he', 'were', 'its', 'formerly', 'now', 'through'}

# stop_words = set(['a', 'able', 'about', 'above', 'across', 'again', "ain't", 'all', 'almost', 'along', 'also', 'am', 'among', 'amongst', 'an', 'and', 'anyhow', 'anyone', 'anyway', 'anyways', 'appear', 'are', 'around', 'as', "a's", 'aside', 'ask', 'asking', 'at', 'away', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'behind', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'came', 'can', 'come', 'comes', 'consider', 'considering', 'corresponding', 'could', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'etc', 'even', 'ever', 'every', 'ex', 'few', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'from', 'further', 'furthermore', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'happens', 'has', 'have', 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', "here's", 'hereupon', 'hers', 'herself', "he's", 'hi', 'him', 'himself', 'his', 'how', 'hows', 'i', "i'd", 'ie', 'if', "i'll", "i'm", 'in', 'inc', 'indeed', 'into', 'inward', 'is', 'it', "it'd", "it'll", 'its', "it's", 'itself', "i've", 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'lately', 'later', 'latter', 'latterly', 'lest', 'let', "let's", 'looking', 'looks', 'ltd', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'might', 'most', 'my', 'myself', 'name', 'namely', 'nd', 'near', 'nearly', 'need', 'needs', 'neither', 'next', 'nine', 'no', 'non', 'now', 'nowhere', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'per', 'placed', 'que', 'quite', 're', 'regarding', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'seven', 'several', 'she', "she'd", "she'll", "she's", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'that', 'thats', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', "there's", 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'those', 'though', 'three', 'through', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', "t's", 'twice', 'two', 'un', 'under', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'usually', 'value', 'various', 'very', 'via', 'viz', 'vs', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", 'went', 'were', "we're", "weren't", "we've", 'what', 'whatever', "what's", 'when', 'whence', 'whenever', "when's", 'where', 'whereafter', 'whereas', 'whereby', 'wherein', "where's", 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', "who's", 'whose', 'why', "why's", 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'would', "wouldn't", 'yes', 'yet', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"])

sentiment_pos = "Pos"
sentiment_neg = "Neg"
auth_true = "True"
auth_neg = "Fake"


def pprint(collection):
    if isinstance(collection, dict):
        for line in collection.items():
            print(line)
    else:
        for line in collection:
            print(line)


def encode_class(cls):
    if cls in {sentiment_pos, auth_true}:
        return 1
    if cls in {sentiment_neg, auth_neg}:
        return -1
    raise Exception("not in seen classes")

def decode_sent_class(cls):
    if cls == 1:
        return "Pos"
    return "Neg"

def decode_auth_class(cls):
    if cls == 1:
        return "True"
    return "Fake"


def break_train_data_line(text: str) -> Optional[Tuple[str, str, int, int]]:
    if not text:
        return None
    split = text.split(" ")
    review_id, authenticity, sentiment = split[:3]
    review_text = " ".join(split[3:])
    return review_id, review_text, encode_class(authenticity), encode_class(sentiment)


def break_test_data_line(text: str) -> Optional[Tuple[str, str]]:
    if not text:
        return None
    review_id, *review_text = text.split(" ")
    return review_id, " ".join(review_text)


def read_train_data(filename: str) -> List[Tuple[str, str, int, int]]:
    train_data = []
    fp = open(filename, 'r')
    for text_line in fp.read().splitlines():
        train_data.append(break_train_data_line(text_line))
    fp.close()
    return train_data


def read_test_data(filename: str) -> List[Tuple[str, str]]:
    fp = open(filename, 'r')
    test_data = []
    for line in fp.read().splitlines():
        test_data.append(break_test_data_line(line))
    fp.close()
    return test_data


def read_dev_data(filename: str) -> List[Tuple[str, str]]:
    dev_data = []
    fp = open(filename, 'r')
    for line in fp.read().splitlines():
        review_id, *review_text = line.split(" ")
        dev_data.append((review_id, " ".join(review_text)))
    fp.close()
    return dev_data


def read_dev_key_data(filename) -> Dict[str, Tuple[int, int]]:
    fp = open(filename)
    dev_key_map = {}
    for line in fp.read().splitlines():
        review_id, authentic, sentiment = line.split(" ")
        dev_key_map[review_id] = (encode_class(authentic), encode_class(sentiment))
    fp.close()
    return dev_key_map


def write_predictions(predictions, filename):
    buffer = []
    for (review_id, auth, sent) in predictions:
        buffer.append(" ".join([review_id, auth, sent]))

    with open(filename, 'w') as fp:
        fp.write("\n".join(buffer))


def get_performance_measure(prediction: List[int], gold: List[int], cls: int):
    total = len(prediction)
    true_positive = 0
    pos_cls_gold = len(list(filter(lambda x: x == cls, gold)))
    pos_cls_pred = len(list(filter(lambda x: x == cls, prediction)))
    for pred, gold in zip(prediction, gold):
        if pred == cls and pred == gold:
            true_positive += 1
    pos_cls_pred = max(1, pos_cls_pred)
    precision = true_positive / pos_cls_pred
    recall = true_positive / pos_cls_gold
    f1 = 2 / ((1 / precision) + (1 / recall))
    return precision, recall, f1


def prepare_train(data):
    sentiment_data = []
    authenticity_data = []
    text = []
    for reviewId, reviewText, auth, sent in data:
        sentiment_data.append((sent, reviewText))
        authenticity_data.append((auth, reviewText))
        text.append(reviewText)
    return sentiment_data, authenticity_data, text

def prepare_train_data(filename: str):
    return prepare_train(read_train_data(filename))


def dev_decode(pred_dict, gold_dict, i):
    gold = []
    predictions = []
    for (reviewId, pred) in pred_dict.items():
        predictions.append(pred)
        gold.append(gold_dict[reviewId][i])
    return predictions, gold


def get_cbow(text: List[str], use_bigrams: bool) -> Dict[str, int]:
    bag_of_words = defaultdict(int)
    for line in text:
        for word in get_text_ngrams(line, False, use_bigrams):
            bag_of_words[word] += 1
    return dict(bag_of_words)


def get_text_ngrams(text: str, remove_stop_words: bool, bigrams: bool = False) -> List[str]:
    if bigrams:
        unigrams = [word.strip() for word in text.lower().split(" ")]
        if remove_stop_words:
            return [word for word in unigrams if word not in stop_words]
        sentences = text.lower().split(".")
        bigrams = []
        for sentence in sentences:
            sentence_unigrams = sentence.split(" ")
            for (u1, u2) in zip(sentence_unigrams, sentence_unigrams[1:]):
                bigrams.append("{} {}".format(u1, u2))

        features = unigrams + bigrams
        return features


    unigrams = [word.strip() for word in text.lower().split(" ")]
    if remove_stop_words:
        return [word for word in unigrams if word not in stop_words]
    return unigrams


def get_sentiment_features(text: str) -> Dict[str, int]:
    return Counter(get_text_ngrams(text, True, False))


def get_authenticity_features(text: str) -> Dict[str, int]:
    return Counter(get_text_ngrams(text, False, True))


def write_output(data: List[str]):
    file = "{}/{}".format(os.getcwd(), output_filename)
    with open(file, 'w') as fp:
        fp.write("\n".join(data))


def write_model(data, filename):
    file = "{}/{}".format(os.getcwd(), filename)
    with open(file, 'w') as fp:
        json.dump(data, fp)

def read_model(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)


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

if __name__ == '__main__':
    model_filename = sys.argv[1]
    data_filename = sys.argv[2]
    model_predict(model_filename, data_filename)
