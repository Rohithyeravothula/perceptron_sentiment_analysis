from statistics import mean

from util import break_train_data_line, full_data_filename, prepare_train, get_cbow, \
    get_sentiment_features, get_authenticity_features, get_performance_measure, dev_decode
from random import shuffle
from perceptron import model_build, model_predict_data, Vanilla_Perceptron, train_and_predict, \
    Avg_Perceptron


def read_full_data(filename):
    full_data = []
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            full_data.append(break_train_data_line(line))
    return full_data


def random_split_test_train(data):
    shuffle(data)
    data_length = len(data)
    split_index = int(0.75*data_length)
    test_data, test_data_key = convert_to_test(data[split_index:])
    return data[0:split_index], test_data, test_data_key

def convert_to_test(data):
    test_data = []
    test_data_key = {}
    for line in data:
        # print(line)
        review_id, review_text, auth, sent = line
        test_data.append((review_id, review_text))
        test_data_key[review_id] = (auth, sent)
    return test_data, test_data_key


def k_fold_test(k: int):
    data = read_full_data(full_data_filename)
    k_fold_iterations = []
    for i in range(k):
        train_data, test_data, test_key = random_split_test_train(data)
        sentiment_data, authenticity_data, text = prepare_train(train_data)
        bag_of_words = get_cbow(text)


        vp_sent = Vanilla_Perceptron(bag_of_words, 0, True)
        vp_auth = Vanilla_Perceptron(bag_of_words, 0, False)

        vp_sent_predictions = train_and_predict(vp_sent, sentiment_data, test_data, get_sentiment_features)
        vp_auth_predictions = train_and_predict(vp_auth, authenticity_data, test_data, get_authenticity_features)

        vp_sent_predictions, sent_gold = dev_decode(vp_sent_predictions, test_key, 1)
        vp_auth_predictions, auth_gold = dev_decode(vp_auth_predictions, test_key, 0)


        _, _, vp_sent_pos_f1 = get_performance_measure(vp_sent_predictions, sent_gold, 1)
        _, _, vp_sent_neg_f1 = get_performance_measure(vp_sent_predictions, sent_gold, -1)
        _, _, vp_auth_pos_f1 = get_performance_measure(vp_auth_predictions, auth_gold, 1)
        _, _, vp_auth_neg_f1 = get_performance_measure(vp_auth_predictions, auth_gold, -1)

        vp_performance = [vp_sent_pos_f1, vp_sent_neg_f1, vp_auth_pos_f1, vp_auth_neg_f1]






        ap_sent = Avg_Perceptron(bag_of_words, 0, True)
        ap_auth = Avg_Perceptron(bag_of_words, 0, False)

        avg_sent_predictions = train_and_predict(ap_sent, sentiment_data, test_data, get_sentiment_features)
        avg_auth_predictions = train_and_predict(ap_auth, authenticity_data, test_data, get_authenticity_features)

        avg_sent_predictions, sent_gold = dev_decode(avg_sent_predictions, test_key, 1)
        avg_auth_predictions, auth_gold = dev_decode(avg_auth_predictions, test_key, 0)

        _, _, avg_sent_pos_f1 = get_performance_measure(avg_sent_predictions, sent_gold, 1)
        _, _, avg_sent_neg_f1 = get_performance_measure(avg_sent_predictions, sent_gold, -1)
        _, _, avg_auth_pos_f1 = get_performance_measure(avg_auth_predictions, auth_gold, 1)
        _, _, avg_auth_neg_f1 = get_performance_measure(avg_auth_predictions, auth_gold, -1)

        avg_performance = [avg_sent_pos_f1, avg_sent_neg_f1, avg_auth_pos_f1, avg_auth_neg_f1]

        # print(vp_performance)
        # print(avg_performance)
        m1, m2 = mean(vp_performance), mean(avg_performance)
        print(m1, m2)
        k_fold_iterations.append((m1, m2))
    m1_mean = mean([m1 for (m1, m2) in k_fold_iterations])
    m2_mean = mean([m2 for (m1, m2) in k_fold_iterations])
    print("behold the final mean")
    print(m1_mean, m2_mean)

k_fold_test(10)
