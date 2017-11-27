import numpy as np
from sklearn.preprocessing import MinMaxScaler


def transform_to_labeled_feature_vector(labeled_data):

    x = np.array(transform_to_feature_vector(labeled_data))
    y = np.array(list(map(lambda r: int(r['is_match']), labeled_data)))

    return x, y


def transform_to_feature_vector(data):
    return list(map(lambda r: [r['tfidf_cosine_similarity'],
                               r['tfidf_cosine_similarity_abt_name_only']], data))
    # return list(map(lambda r: r['word_similarities_vector'], data))


def map_predictions_to_data(predictions, data):
    for idx in range(len(data)):
        data[idx]['is_match_prediction'] = True if predictions[idx] == 1 else False

    return data
