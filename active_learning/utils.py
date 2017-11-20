import numpy as np
from sklearn.preprocessing import MinMaxScaler


def transform_to_labeled_feature_vector(labeled_data):
    # x = []
    # y = []
    #
    # for idx in idx_list:
    #     x.append([data[idx]['edit_similarity'], data[idx]['tfidf_cosine_similarity']])
    #     y.append(int(data[idx]['is_match']))

    x = np.array(transform_to_feature_vector(labeled_data))
    y = np.array(list(map(lambda r: int(r['is_match']), labeled_data)))

    return x, y


def transform_to_feature_vector(data):
    # return list(map(lambda r: [r['word_similarities_vector'],
    #                            r['monge_elkan_similarity'],
    #                            r['generalized_jaccard_similarity']], data))
    return list(map(lambda r: r['word_similarities_vector'], data))


def map_predictions_to_data(predictions, data):
    for idx in range(len(data)):
        data[idx]['is_match_prediction'] = True if predictions[idx] == 1 else False

    return data
