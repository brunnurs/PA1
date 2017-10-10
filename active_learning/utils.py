def transform_to_labeled_feature_vector(labeled_data):
    # x = []
    # y = []
    #
    # for idx in idx_list:
    #     x.append([data[idx]['edit_similarity'], data[idx]['tfidf_cosine_similarity']])
    #     y.append(int(data[idx]['is_match']))

    x = transform_to_feature_vector(labeled_data)
    y = list(map(lambda r: int(r['is_match']), labeled_data))

    return x, y


def transform_to_feature_vector(data):
    return list(map(lambda r: [r['edit_similarity'], r['tfidf_cosine_similarity']], data))


def map_predictions_to_data(predictions, data):
    for idx in range(len(data)):
        data[idx]['is_match_prediction'] = True if predictions[idx] == 1 else False

    return data
