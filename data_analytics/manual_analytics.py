import pickle

from py_stringmatching import Jaro, SoftTfIdf

from preprocessing.preprocessing import bag_of_words
import matplotlib.pyplot as plt


def _transform_to_clean_string(token_array):
    return ' '.join(token_array)


def _plot_histogram_word_length(data):
    lengths_bag_of_words = list(map(lambda r: len(r['record_b']['bag_of_words']), data))

    binwidth = 5
    bins = range(min(lengths_bag_of_words), max(lengths_bag_of_words) + binwidth, binwidth)

    n, bins, _ = plt.hist(lengths_bag_of_words, bins)
    plt.xlabel('length of bag of words (binned)')
    plt.ylabel('amount of samples in each bin')
    plt.show()


def tf_idf_with_corpse():
    with open('tf_idf_similarity_jaro_95', 'rb') as handle:
        tf_idf_similarity = pickle.load(handle)

        abt = 'mavis beacon typing 17 standard by broderbund encore software'
        buy = 'encore software encore software mavis beacon typing 17 standardby broderbund'

        print(tf_idf_similarity.calculate_similarity(bag_of_words(abt), bag_of_words(buy)))

    # with open('tf_idf_similarity_jaro_95_name_only', 'rb') as handle:
    #     tf_idf_similarity = pickle.load(handle)
    #
    #     abt = 'route 66 mobile 7 usa cdn for windows mobile 5'
    #     buy = 'route 66 mobile 7 usa cdn for windows mobile 5'
    #
    #     print(tf_idf_similarity.calculate_similarity(bag_of_words(abt), bag_of_words(buy)))



def jaro_simple_word_similarity():
    print(Jaro().get_sim_score('cli221gry', 'cli221'))


def manual_analytics():
    with open("{}/{}".format('tf_idf_only', 'good_results'), 'rb') as handle:
        good_results = pickle.load(handle)

    with open("{}/{}".format('tf_idf_only', 'bad_results'), 'rb') as handle:
        bad_results = pickle.load(handle)

    # _plot_histogram_word_length(bad_results)

    print('========================= good results: {} ========================='.format(len(good_results)))
    for good_result in good_results:
        print('record-a-id: {} record-b-id:{}'.format(
            good_result['record_a']['record_id'],
            good_result['record_b']['record_id'])
        )

        print(_transform_to_clean_string(good_result['record_a']['bag_of_words']))
        print(_transform_to_clean_string(good_result['record_b']['bag_of_words']))
        print('tf/idf description: {}'.format(good_result['tfidf_cosine_similarity']))
        print()
        print(_transform_to_clean_string(good_result['record_a']['bag_of_words_name']))
        print(_transform_to_clean_string(good_result['record_b']['bag_of_words_name']))
        print('tf/idf name: {}'.format(good_result['tfidf_cosine_similarity_name_only']))
        print()
        print(_transform_to_clean_string(good_result['record_a']['bag_of_words_name'] + good_result['record_a']['bag_of_words']))
        print(_transform_to_clean_string(good_result['record_b']['bag_of_words_name'] + good_result['record_b']['bag_of_words']))
        print('tf/idf combined: {}'.format(good_result['tfidf_cosine_similarity_combined']))
        print()
        print()

    print()
    print()
    print('========================= bad results: {} ========================='.format(len(bad_results)))

    for bad_result in bad_results:
        print('record-a-id: {} record-b-id:{}'.format(
            bad_result['record_a']['record_id'],
            bad_result['record_b']['record_id'])
        )

        print(_transform_to_clean_string(bad_result['record_a']['bag_of_words']))
        print(_transform_to_clean_string(bad_result['record_b']['bag_of_words']))
        print('tf/idf description: {}'.format(bad_result['tfidf_cosine_similarity']))
        print()
        print(_transform_to_clean_string(bad_result['record_a']['bag_of_words_name']))
        print(_transform_to_clean_string(bad_result['record_b']['bag_of_words_name']))
        print('tf/idf name: {}'.format(bad_result['tfidf_cosine_similarity_name_only']))
        print()
        print(_transform_to_clean_string(bad_result['record_a']['bag_of_words_name'] + good_result['record_a']['bag_of_words']))
        print(_transform_to_clean_string(bad_result['record_b']['bag_of_words_name'] + good_result['record_b']['bag_of_words']))
        print('tf/idf combined: {}'.format(bad_result['tfidf_cosine_similarity_combined']))
        print()
        print()
        print()


def save_information_about_predictions(clf, x_test, y_test, indices_test, data):
    good_results = []
    bad_results = []

    for idx, x_value in enumerate(x_test):
        if y_test[idx] == 1 and clf.predict([x_value]) == 1:
            good_results.append(data[indices_test[idx]])
        elif y_test[idx] == 1 and clf.predict([x_value]) == 0:
            bad_results.append(data[indices_test[idx]])

    with open("{}/{}".format('data_analytics/tf_idf_only', 'good_results'), 'wb') as handle:
        pickle.dump(good_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("{}/{}".format('data_analytics/tf_idf_only', 'bad_results'), 'wb') as handle:
        pickle.dump(bad_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    manual_analytics()
    # tf_idf_with_corpse()
    # jaro_simple_word_similarity()
