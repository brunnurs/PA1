import numpy as np
from sklearn.model_selection import train_test_split


def stratified_random_indices(data, number_of_random_indices):
    """
    Return a certain amount of indices randomly. THe data array is used for the range.
    Important: the result is stratified, so it contains the same percentage of a certain class as all the other indices.
    :param data:
    :param number_of_random_indices:
    :return:
    """
    x = np.array(transform_to_feature_vector(data))
    y = np.array(list(map(lambda r: int(r['ground_truth']), data)))

    indices = range(len(x))

    _, _, _, _, indices_train, _ = train_test_split(x, y, indices, train_size=number_of_random_indices, stratify=y)

    return indices_train


def transform_to_labeled_feature_vector(labeled_data):

    x = np.array(transform_to_feature_vector(labeled_data))
    y = np.array(list(map(lambda r: int(r['is_match']), labeled_data)))

    return x, y


def transform_to_feature_vector(data):
    return list(map(lambda r: [r['tfidf_cosine_similarity'],
                               r['tfidf_cosine_similarity_name_only']], data))

    # return list(map(lambda r: r['word_similarities_vector'], data))


def map_predictions_to_data(predictions, data):
    for idx in range(len(data)):
        data[idx]['is_match_prediction'] = True if predictions[idx] == 1 else False

    return data
