"""
Code for loading different datasets in a unified format

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import csv
import h5py
import io
import numpy as np
import pandas as pd
import pkgutil
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(dataset, random_seed_offset=0, test_size=0.3,
                 fix_test_shuffle_train=False, competing=False,
                 time_series=False):
    """
    Loads a dataset.

    Parameters
    ----------
    dataset : string
        For this demo, one of 'rotterdam-gbsg' or 'support'.

        Note that a dataset could come with its own pre-specified training
        and test data (for example, for 'rotterdam-gbsg', we train
        on the Rotterdam tumor bank data and test on GBSG2 data). If the
        dataset does *not* come with pre-specified training and test data
        (such as for 'support'), then the code will generate a "random"
        train/test split ("random" is in quotes since to make the code
        reproducible, we have seeded the randomness per dataset with a fixed
        integer, so that the code should produce the same random train/test
        splits across experimental repeats when
        `random_seed_offset` is set to the same value).

    random_seed_offset : int, optional (default=0)
        Offset to add to random seed in shuffling the data.

    test_size: float or int, optional (default=0.3)
        If specified as a float, then this is the fraction of data to treat as
        the test data. If specified as an int, then this is the number of
        points to treat as the test data.

    fix_test_shuffle_train: boolean (default=False)
        If this flag is set to be True, then the test data will be treated as
        fixed meaning that using different values of `random_seed_offset` lead
        to different shuffled versions of the training data but the test data
        remain the same.

        If instead this flag is set to be False, then we actually shuffle the
        full dataset prior to making the train/test split. Thus, different
        values of `random_seed_offset` lead to different train/test splits
        (so which points end up in training vs test data varies across
        different values of `random_seed_offset`).

    competing: boolean (default=False)
        If this flag is set to be True, then the training and test set labels
        will use event indicators that specify which critical event happened
        (where 0 means censoring happened prior to any critical event).
        
        Note that if the dataset does not actually support competing risks,
        then the value of this flag does not affect anything (basically we
        would just be using the competing risks setup with a single critical
        event).

    time_series: boolean (default=False)
        For datasets that are not actually time series, this flag should be
        left as the default value of False (and setting this flag to be True
        for such datasets will result in an error).

        For datasets that are time series, if this flag is set to be False,
        then we only take the first time step per time series, and we thus
        treat the dataset as if it were a regular tabular dataset. If
        instead, this flag is set to be True for a time series dataset,
        then the training and test data can be variable in length.

    Returns
    -------
    X_train : either a 2D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 2D numpy array with shape = [n_samples, n_features].
        This array will consist of the training feature vectors.

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series represented as a 2D numpy array with
        shape [n_time_steps, n_features]. Different data points could have
        different numbers of time steps.

    Y_train : either a 1D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 1D numpy array with length = n_samples. This array
        will consist of training observed times (in the same order as the
        rows of `X_train`).

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series's observed times across time, represented as
        a 1D numpy array with length = n_time_steps.

    D_train : either a 1D numpy array or a list
        If we are working with tabular data (`time_series` is False), then
        this will be a 1D numpy array with length = n_samples. This array
        will consist of training event indicators (in the same order as the
        rows of `X_train`).

        If we are working with time series data (`time_series` is True), then
        this will be a list. The i-th entry in the list corresponds to the
        i-th training time series's event indicators across time (which
        commonly does not actually change), represented as a 1D numpy array
        with length = n_time_steps.

    X_test : either a 2D numpy array or a list
        Same format as `X_train` but now for test data.

    Y_test : either a 1D numpy array or a list
        Same format as `Y_train` but now for test data.

    D_test : either a 1D numpy array or a list
        Same format as `D_train` but now for test data.

    features_before_preprocessing : list
        List of strings specifying the names of the features *before* applying
        preprocessing.

    features_after_preprocessing : list
        List of strings specifying the names of the features *after* applying
        preprocessing.

    events : list
        List of strings specifying the event names. For standard survival
        analysis datasets (without competing risks), this list will consist of
        a single element. When there are competing risks, the length of this
        list is the number of competing risks.

    train_test_split_prespecified : boolean
        If True, then this means that the dataset comes with its own train/test
        split and therefore there is no randomization for this split.

    build_preprocessor_and_preprocess : function
        Function for fitting and then preprocessing features into some
        "standardized"/"normalized" feature space. This should be applied to
        training feature vectors prior to using a learning algorithm (unless the
        learning algorithm does not need this sort of normalization). This
        function returns both the normalized features and a preprocessor object
        (see the next output for how to use this preprocessor object).

    apply_preprocessor : function
        Function that, given feature vectors (e.g., validation/test data) and a
        preprocessor object (created via `build_preprocessor_and_preprocess`),
        preprocesses the feature vectors as to put them in a normalized feature
        space.
    """
    if dataset == 'support':
        if time_series:
            raise Exception('The SUPPORT dataset is not longitudinal')

        raw_df = pd.read_csv('data/support2.csv')

        # we use the same features as in the DeepSurv paper (Katzman et al., 2018)

        # the preprocessing we do below results in the same total number of
        # subjects as in the DeepSurv paper's SUPPORT dataset (8,873)
        df = raw_df[['age', 'sex', 'race', 'num.co', 'diabetes',
                     'dementia', 'ca', 'meanbp', 'hrt', 'resp',
                     'temp', 'wblc', 'sod', 'crea',
                     'd.time', 'death']].astype({'race': 'category'})

        def map_cancer_to_int(x):
            if x == 'metastatic':
                return 2
            elif x == 'yes':
                return 1
            else:
                return 0

        df['sex'] = df['sex'].apply(lambda x: 1 * (x == 'female'))
        df = df.rename(columns={'sex': 'female'})
        df['ca'] = df['ca'].apply(map_cancer_to_int)

        race_categories = ['blank'] + df['race'].cat.categories.to_list()
        df['race'] = df['race'].cat.codes + 1

        df = df.dropna()

        features_before_preprocessing = ['age', 'female', 'race', 'num.co', 'diabetes',
                                         'dementia', 'ca', 'meanbp', 'hrt', 'resp',
                                         'temp', 'wblc', 'sod', 'crea']

        X = df[features_before_preprocessing].to_numpy().astype('float32')
        Y = df['d.time'].to_numpy().astype('float32')
        D = df['death'].to_numpy().astype('int32')

        features_after_preprocessing = \
            ['age_std', 'female', 'num.co_norm', 'diabetes', 'dementia',
             'ca_norm', 'meanbp_std', 'hrt_std', 'resp_std', 'temp_std',
             'wblc_std', 'sod_std', 'crea_std',
             'race_blank', 'race_asian', 'race_black',
             'race_hispanic', 'race_other', 'race_white']

        events = ['death']

        categories = [list(range(int(X[:, 2].max()) + 1))]

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
               0: age       (we will standardize, i.e., compute Z score)
               1: female    (we will leave the same; originally a binary indicator)
               2: race      (we will treat as categorical and one-hot encode)
               3: num_co    (we will divide by 9 as a simple normalization)
               4: diabetes  (we will leave the same; originally a binary indicator)
               5: dementia  (we will leave the same; originally a binary indicator)
               6: ca        (we will divide by 2 as a simple normalization)
               7: meanbp    (we will standardize)
               8: hrt       (we will standardize)
               9: resp      (we will standardize)
              10: temp      (we will standardize)
              11: wblc      (we will standardize)
              12: sod       (we will standardize)
              13: crea      (we will standardize)

            After preprocessing, the new features are:
               0: standardized age
               1: female
               2: num_co / 9
               3: diabetes
               4: dementia
               5: ca / 2
               6: standardized meanbp
               7: standardized hrt
               8: standardized resp
               9: standardized temp
              10: standardized wblc
              11: standardized sod
              12: standardized crea
              13: race left blank
              14: race = asian
              15: race = black
              16: race = hispanic
              17: race = other
              18: race = white

            When a Cox model is fitted (argument `cox` is set to True),
            then we drop feature 18 treating it as the reference value
            for categorical variable `race` as to avoid collinearity
            (this sort of preprocessing is standard for working with
            Cox models and categorical variables where one category
            is treated as the reference or baseline value and is omitted).
            """
            new_features = np.zeros((features.shape[0], 19))
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=categories)
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.fit_transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.fit_transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            if cox:
                return new_features[:, :-1], (scaler, encoder)
            return new_features, (scaler, encoder)

        def apply_preprocessor(features, preprocessor, cox=False):
            new_features = np.zeros((features.shape[0], 19))
            scaler, encoder = preprocessor
            cols_standardize = [0, 7, 8, 9, 10, 11, 12, 13]
            cols_leave = [1, 4, 5]
            cols_categorical = [2]
            new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
                scaler.transform(features[:, cols_standardize])
            new_features[:, [1, 3, 4]] = features[:, cols_leave]
            new_features[:, 13:] = \
                encoder.transform(features[:, cols_categorical]).toarray()
            new_features[:, 2] = features[:, 3] / 9.
            new_features[:, 5] = features[:, 6] / 2.
            if cox:
                return new_features[:, :-1]
            return new_features

        dataset_random_seed = 331231101
        train_test_split_prespecified = False

    elif dataset == 'rotterdam-gbsg':
        if time_series:
            raise Exception('The Rotterdam/GBSG datasets are not longitudinal')

        # ----------------------------------------------------------------------
        # snippet of code from DeepSurv repository
        datasets = defaultdict(dict)
        with h5py.File('data/gbsg_cancer_train_test.h5', 'r') as fp:
            for ds in fp:
                for array in fp[ds]:
                    datasets[ds][array] = fp[ds][array][:]
        # ----------------------------------------------------------------------

        features_before_preprocessing \
            = ['horTh', 'tsize', 'menostat', 'age', 'pnodes',
               'progrec', 'estrec']

        features_after_preprocessing \
            = ['horTh', 'tsize_norm', 'menostat', 'age_std', 'pnodes_std',
               'progrec_std', 'estrec_std']

        events = ['death']

        X_train = datasets['train']['x'].astype('float32')
        Y_train = datasets['train']['t'].astype('float32')
        D_train = datasets['train']['e'].astype('int32')
        X_test = datasets['test']['x'].astype('float32')
        Y_test = datasets['test']['t'].astype('float32')
        D_test = datasets['test']['e'].astype('int32')

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
              0: horTh     (we will leave the same)
              1: tsize     (we will divide by 2)
              2: menostat  (we will leave the same)
              3: age       (we will standardize, i.e., compute Z score)
              4: pnodes    (we will standardize)
              5: progrec   (we will standardize)
              6: estrec    (we will standardize)

            After preprocessing, the new features are:
              0: horTh
              1: tsize / 2
              2: menostat
              3: standardized age
              4: standardized pnodes
              5: standardized progrec
              6: standardized estrec
            """
            new_features = np.zeros_like(features)
            preprocessor = StandardScaler()
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                preprocessor.fit_transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features, preprocessor

        def apply_preprocessor(features, preprocessor, cox=False):
            new_features = np.zeros_like(features)
            cols_standardize = [3, 4, 5, 6]
            cols_leave = [0, 2]
            new_features[:, cols_standardize] = \
                preprocessor.transform(features[:, cols_standardize])
            new_features[:, cols_leave] = features[:, cols_leave]
            new_features[:, 1] = features[:, 1] / 2.
            return new_features

        dataset_random_seed = 1831262265
        train_test_split_prespecified = True  # since we train on rotterdam and test on gbsg

    elif dataset == 'pbc':
        df = pd.read_csv('data/pbc2.csv').astype({'edema': 'category'})

        def map_yes_no_or_missing_to_number(x):
            if type(x) != str:
                return np.nan
            elif x == 'Yes':
                return 1.0
            else:
                return 0.0

        df['drug'] = df['drug'].apply(lambda x: 1*(x == 'D-penicil'))  # no nan
        df['sex'] = df['sex'].apply(lambda x: 1*(x == 'female'))  # no nan
        df['ascites'] = df['ascites'].apply(map_yes_no_or_missing_to_number)
        df['hepatomegaly'] = \
            df['hepatomegaly'].apply(map_yes_no_or_missing_to_number)
        df['spiders'] = df['spiders'].apply(map_yes_no_or_missing_to_number)
        df = df.rename(columns={'sex': 'female', 'drug': 'D-penicil'})
        df['age'] = df['age'] + df['years']
        edema_categories = df['edema'].cat.categories.to_list()
        df['edema'] = df['edema'].cat.codes

        features_before_preprocessing = \
            ['D-penicil', 'female', 'ascites', 'hepatomegaly', 'spiders',
             'edema', 'histologic', 'serBilir', 'serChol', 'albumin',
             'alkaline', 'SGOT', 'platelets', 'prothrombin', 'age']

        features_after_preprocessing = \
            ['D-penicil', 'female', 'ascites', 'hepatomegaly', 'spiders',
             'histologic_norm', 'serBilir_std', 'serChol_std', 'albumin_std',
             'alkaline_std', 'SGOT_std', 'platelets_std', 'prothrombin_std',
             'age_std', 'edema_no', 'edema_yes_despite_diuretics',
             'edema_yes_without_diuretics']

        features = df[features_before_preprocessing].to_numpy().astype('float32')
        observed_times = (df['years'] - df['year']).to_numpy().astype('float32')
        event_indicators = (df['status'] == 'dead').to_numpy().astype('int32')
        events = ['death']
        if competing:
            event_indicators[df['status'] == 'transplanted'] = 2
            events.append('transplanted')

        if not time_series:
            X = features
            Y = observed_times
            D = event_indicators
        else:
            X = []
            Y = []
            D = []
            for id in sorted(list(set(df['id']))):
                mask = (df['id'] == id)
                X.append(features[mask])
                Y.append(observed_times[mask])
                D.append(event_indicators[mask])

        def build_preprocessor_and_preprocess(features, cox=False):
            """
            Prior to preprocessing, the features are expected to be:
               0: D-penicil    (we will leave the same; originally a binary indicator)
               1: female       (we will leave the same; originally a binary indicator)
               2: ascites      (we will leave the same; originally a binary indicator)
               3: hepatomegaly (we will leave the same; originally a binary indicator)
               4: spiders      (we will leave the same; originally a binary indicator)
               5: edema        (we will treat as categorical and one-hot encode)
               6: histologic   (we will subtract 1 and then divide by 3)
               7: serBilir     (we will standardize)
               8: serChol      (we will standardize)
               9: albumin      (we will standardize)
              10: alkaline     (we will standardize)
              11: SGOT         (we will standardize)
              12: platelets    (we will standardize)
              13: prothrombin  (we will standardize)
              14: age          (we will standardize)

            After preprocessing, the new features are:
               0: D-penicil
               1: female
               2: ascites
               3: hepatomegaly
               4: spiders
               5: (histologic - 1) / 3
               6: standardized serBilir
               7: standardized serChol
               8: standardized albumin
               9: standardized alkaline
              10: standardized SGOT
              11: standardized platelets
              12: standardized prothrombin
              13: standardized age
              14: edema = no
              15: edema = yes despite diuretics
              16: edema = yes without diuretics

            When a Cox model is fitted (argument `cox` is set to True),
            then we drop feature 18 treating it as the reference value
            for categorical variable `race` as to avoid collinearity
            (this sort of preprocessing is standard for working with
            Cox models and categorical variables where one category
            is treated as the reference or baseline value and is omitted).
            """
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 17))
            imputer = SimpleImputer(missing_values=np.nan,
                                    strategy='mean')
            scaler = StandardScaler()
            encoder = OneHotEncoder(categories=[list(range(3))])
            features_imputed = imputer.fit_transform(features_stacked)
            cols_standardize = [7, 8, 9, 10, 11, 12, 13, 14]
            cols_leave = [0, 1, 2, 3, 4]
            cols_categorical = [5]
            new_features[:, [0, 1, 2, 3, 4]] \
                = features_imputed[:, cols_leave]
            new_features[:, 5] = (features_imputed[:, 6] - 1) / 3.
            new_features[:, [6, 7, 8, 9, 10, 11, 12, 13]] = \
                scaler.fit_transform(
                    features_imputed[:, cols_standardize])
            new_features[:, 14:] = \
                encoder.fit_transform(
                    features_imputed[:, cols_categorical]).toarray()
            if cox:
                new_features = new_features[:, :-1]

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked, (imputer, scaler, encoder)
            else:
                return new_features, (imputer, scaler, encoder)

        def apply_preprocessor(features, preprocessor, cox=False):
            if type(features) == list:
                features_stacked = np.vstack(features)
                lengths = [len(_) for _ in features]
            else:
                features_stacked = features

            new_features = np.zeros((features_stacked.shape[0], 17))
            imputer, scaler, encoder = preprocessor
            features_imputed = imputer.transform(features_stacked)
            cols_standardize = [7, 8, 9, 10, 11, 12, 13, 14]
            cols_leave = [0, 1, 2, 3, 4]
            cols_categorical = [5]
            new_features[:, [0, 1, 2, 3, 4]] \
                = features_imputed[:, cols_leave]
            new_features[:, 5] = (features_imputed[:, 6] - 1) / 3.
            new_features[:, [6, 7, 8, 9, 10, 11, 12, 13]] = \
                scaler.transform(
                    features_imputed[:, cols_standardize])
            new_features[:, 14:] = \
                encoder.transform(
                    features_imputed[:, cols_categorical]).toarray()
            if cox:
                new_features = new_features[:, :-1]

            if type(features) == list:
                idx = 0
                features_unstacked = []
                for length in lengths:
                    features_unstacked.append(
                        new_features[idx:idx+length])
                    idx += length
                return features_unstacked
            else:
                return new_features

        dataset_random_seed = 2893429804
        train_test_split_prespecified = False

    else:
        raise NotImplementedError('Unsupported dataset: %s' % dataset)

    if train_test_split_prespecified:
        # this first case corresponds to when the dataset has pre-specified
        # training and test data (so that the variables `X_train`, `y_train`,
        # `X_test`, and `y_test` have been defined earlier already)

        # shuffle only the training data (do not modify `X_test` and `y_test`
        # that are defined earlier)
        rng = np.random.RandomState(dataset_random_seed + random_seed_offset)
        shuffled_indices = rng.permutation(len(X_train))

        if type(X_train) == np.ndarray:
            X_train = X_train[shuffled_indices]
            Y_train = Y_train[shuffled_indices]
            D_train = D_train[shuffled_indices]
        else:
            assert type(X_train) == list
            X_train = [X_train[idx] for idx in shuffled_indices]
            Y_train = [Y_train[idx] for idx in shuffled_indices]
            D_train = [D_train[idx] for idx in shuffled_indices]
    else:
        # this second case corresponds to when the dataset does *not* have
        # pre-specified training and test data, so we need to define
        # `X_train`, `y_train`, `X_test`, and `y_test` first via a random split
        # (we use sklearn's `train_test_split` function)

        # note that by default, sklearn's `train_test_split` will shuffle the
        # data before doing the split
        if fix_test_shuffle_train:
            # first come up with training and test data
            rng = np.random.RandomState(dataset_random_seed)
            X_train, X_test, Y_train, Y_test, D_train, D_test = \
                train_test_split(X, Y, D, test_size=test_size, random_state=rng)

            if random_seed_offset > 0:
                # do another shuffle of only the training data using the seed
                # offset (at this point, we treat the test set defined by
                # `X_test` and `y_test` as fixed)
                rng = np.random.RandomState(dataset_random_seed
                                            + random_seed_offset)
                shuffled_indices = rng.permutation(len(X_train))

                if type(X_train) == np.ndarray:
                    X_train = X_train[shuffled_indices]
                    Y_train = Y_train[shuffled_indices]
                    D_train = D_train[shuffled_indices]
                else:
                    assert type(X_train) == list
                    X_train = [X_train[idx] for idx in shuffled_indices]
                    Y_train = [Y_train[idx] for idx in shuffled_indices]
                    D_train = [D_train[idx] for idx in shuffled_indices]
        else:
            # in this case, we do not treat the test data as fixed so that
            # different `random_seed_offset` values will yield different
            # train/test splits
            rng = np.random.RandomState(dataset_random_seed
                                        + random_seed_offset)
            X_train, X_test, Y_train, Y_test, D_train, D_test = \
                train_test_split(X, Y, D, test_size=test_size, random_state=rng)

    return X_train, Y_train, D_train, X_test, Y_test, D_test, \
            features_before_preprocessing, features_after_preprocessing, \
            events, train_test_split_prespecified, \
            build_preprocessor_and_preprocess, apply_preprocessor
