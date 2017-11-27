import pickle

from py_stringmatching import Jaro, SoftTfIdf

from preprocessing.preprocessing import bag_of_words


def tf_idf_with_corpse():
    with open('tf_idf_similarity_jaro_95', 'rb') as handle:
        tf_idf_similarity = pickle.load(handle)

        # tf_idf_similarity = SoftTfIdf(sim_func=Jaro().get_raw_score, threshold=0.95)

        abt = 'ge futura indoor tv antenna tv24746 specially designed to receive digital tv signal 20db gain amplification noise eliminator circuitry filter designed to mount horizontally or vertically'
        buy = 'ge 24746 futura tm indoor hdtv antenna ge'

        print(tf_idf_similarity.calculate_similarity(bag_of_words(abt), bag_of_words(buy)))


def jaro_simple_word_similarity():
    print(Jaro().get_sim_score('cli221gry', 'cli221'))


def manual_analytics():
    with open("{}/{}".format('tf_idf_only', 'good_results'), 'rb') as handle:
        good_results = pickle.load(handle)

    with open("{}/{}".format('tf_idf_only', 'bad_results'), 'rb') as handle:
        bad_results = pickle.load(handle)

    min_result_good = 100000
    max_result_good = 0
    average_good = 0

    print('========================= good results: {} ========================='.format(len(good_results)))
    for good_result in good_results:
        print('abt-id: {} buy-id:{}'.format(
            good_result['abt_record']['record_id'],
            good_result['buy_record']['record_id'])
        )

        print(good_result['abt_record']['clean_string'])
        print(good_result['buy_record']['clean_string'])
        print('tf/idf: {}'.format(good_result['tfidf_cosine_similarity']))
        print()

        if good_result['tfidf_cosine_similarity'] < min_result_good:
            min_result_good = good_result['tfidf_cosine_similarity']

        if good_result['tfidf_cosine_similarity'] > max_result_good:
            max_result_good = good_result['tfidf_cosine_similarity']

        average_good += good_result['tfidf_cosine_similarity']

    print('min tf/idf: {}, max tf/idf: {}, avg tf/idf: {}'.format(min_result_good, max_result_good,
                                                                  average_good / len(good_results)))

    print()
    print()
    print('========================= bad results: {} ========================='.format(
        len(bad_results)))

    min_result_bad = 100000
    max_result_bad = 0
    average_bad = 0

    for bad_result in bad_results:
        print('abt-id: {} buy-id:{}'.format(
            bad_result['abt_record']['record_id'],
            bad_result['buy_record']['record_id'])
        )

        print(bad_result['abt_record']['clean_string'])
        print(bad_result['buy_record']['clean_string'])
        print('tf/idf: {}'.format(bad_result['tfidf_cosine_similarity']))
        print()

        if bad_result['tfidf_cosine_similarity'] < min_result_bad:
            min_result_bad = bad_result['tfidf_cosine_similarity']

        if bad_result['tfidf_cosine_similarity'] > max_result_bad:
            max_result_bad = bad_result['tfidf_cosine_similarity']

        average_bad += bad_result['tfidf_cosine_similarity']

    print('min tf/idf: {}, max tf/idf: {}, avg tf/idf: {}'.format(min_result_bad, max_result_bad,
                                                                  average_bad / len(bad_results)))


def save_information_about_predictions(model, x_test, y_test, indices_test, data):
    good_results = []
    bad_results = []

    for idx, x_value in enumerate(x_test):
        if y_test[idx] == 1 and model.predict_classes(x_value) == 1:
            good_results.append(data[indices_test[idx]])
        elif y_test[idx] == 1 and model.predict_classes(x_value) == 0:
            bad_results.append(data[indices_test[idx]])

    with open("{}/{}".format('data_analytics/tf_idf_only', 'good_results'), 'wb') as handle:
        pickle.dump(good_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("{}/{}".format('data_analytics/tf_idf_only', 'bad_results'), 'wb') as handle:
        pickle.dump(bad_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # manual_analytics()
    tf_idf_with_corpse()
    jaro_simple_word_similarity()
