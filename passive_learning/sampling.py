from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.utils import resample

from active_learning.utils import transform_to_labeled_feature_vector


def downsample_to_even_classes(data):
    matches = list(filter(lambda r: r['is_match'], data))
    non_matches = list(filter(lambda r: not r['is_match'], data))

    len_small_class = min(len(matches), len(non_matches))

    x_matches, y_matches = transform_to_labeled_feature_vector(matches)
    x_non_matches, y_non_matches = transform_to_labeled_feature_vector(non_matches)

    x_matches_sampled, y_matches_sampled = resample(x_matches, y_matches, replace=False, n_samples=len_small_class)
    x_non_matches_sampled, y_non_matches_sampled = resample(x_non_matches, y_non_matches, replace=False,
                                                            n_samples=len_small_class)

    # this last resample might be unnecessary... but somehow I don't like both the sets (matches/non-matches) just
    # concatenated
    return resample(x_matches_sampled + x_non_matches_sampled, y_matches_sampled + y_non_matches_sampled, replace=False,
                    n_samples=len_small_class * 2)


def random_oversampling(x, y):
    print('Original dataset shape {}'.format(Counter(y)))

    ros = RandomOverSampler(random_state=42)
    x_sampled, y_sampled = ros.fit_sample(x, y)

    print('With RandomOverSampler sampled dataset shape {}'.format(Counter(y_sampled)))

    return x_sampled, y_sampled


def SMOTE_oversampling(x, y):
    print('Original dataset shape {}'.format(Counter(y)))

    smote = SMOTE(random_state=42)
    x_sampled, y_sampled = smote.fit_sample(x, y)

    print('With SMOTE sampled dataset shape {}'.format(Counter(y_sampled)))

    return x_sampled, y_sampled


def ADASYN_oversampling(x, y):
    print('Original dataset shape {}'.format(Counter(y)))

    adasyn = ADASYN(random_state=42)
    x_sampled, y_sampled = adasyn.fit_sample(x, y)

    ('With ADASYN sampled dataset shape {}'.format(Counter(y_sampled)))

    return x_sampled, y_sampled
