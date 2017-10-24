import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import logging
from utils.logger import logger_initialization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# seed for numpy and sklearn
random_state = 1
np.random.seed(random_state)


# Function to create model, required for KerasClassifier
def create_model():
    """
    Neural Network model
    :return: NN model
    """
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def calculate_show_frequency(train_data, test_data):
    """
    This function will loop through all the unique PATIENT_IDs in the training dataset and calculate the SHOW frequency
    at each new encounter. Then it will go through the testing dataset and it will set the latest SHOW_FREQUENCY of the
    training data point based on the ENCOUNTER_APPOINTMENT_DATETIME
    :param train_data: training dataset, needs two columns 'NOSHOW', 'SHOW_FREQUENCY' and
    'ENCOUNTER_APPOINTMENT_DATETIME'
    :param test_data: testing dataset
    :return: a filled 'SHOW_FREQUENCY' column in the training and testing dataset.
    """

    logging.getLogger('tab.regular').debug('obtaining frequency of "SHOW" and "NOSHOW" based on the patient\'s id')

    unique_training_patient_ids = train_data['PATIENT_KEY'].unique()
    msg = 'there are {0} unique patient IDs in the training dataset.'.format(len(unique_training_patient_ids))
    logging.getLogger('tab.regular').debug(msg)
    logging.getLogger('tab.regular').debug('The first 5 patients, in the training dataset, are: {0}'.format(
        unique_training_patient_ids[:5]))

    # for each patient in the training dataset
    logging.getLogger('tab.regular').debug('Processing patients in the training dataset')
    for patient_key_index, patient_key in enumerate(unique_training_patient_ids):
        msg = 'processing {0} out of {1} patients'.format(patient_key_index, len(unique_training_patient_ids)-1)
        logging.getLogger('tab.tab.regular').debug(msg)

        # get the data point matching the current patient_key
        patient_dataframe = train_data[train_data['PATIENT_KEY'] == patient_key]

        # number of encounters processed
        encounters_processed = 0.0
        # total number of encounter that the patient showed up
        total_shows = 0.0
        # total number of encounter that the patient did not show up
        total_no_shows = 0.0
        # loop through each encounter
        for index, data_point in patient_dataframe.iterrows():
            encounters_processed += 1.0
            # if the patient did not show up
            if data_point['NOSHOW']:
                total_no_shows += 1.0
                prob = 1 - (total_no_shows / encounters_processed)
            else:
                total_shows += 1.0
                prob = total_shows / encounters_processed

            # update the SHOW_FREQUENCY for the specific patient's based on the index processed
            train_data.loc[index, 'SHOW_FREQUENCY'] = prob

    unique_testing_patient_ids = test_data['PATIENT_KEY'].unique()
    msg = 'there are {0} unique patient IDs in the training dataset.'.format(len(unique_testing_patient_ids))
    logging.getLogger('line.tab.regular').debug(msg)
    logging.getLogger('tab.regular').debug('The first 5 patients, in the testing dataset, are: {0}'.format(
        unique_testing_patient_ids[:5]))

    for patient_key in unique_testing_patient_ids:
        # if the patient in the testing dataset is not in the training dataset, then continue to the next patient and
        # do not modified the SHOW_FREQUENCY i.e. leave it to 100% chance of showing up
        if patient_key not in train_data['PATIENT_KEY'].values:
            continue

        training_patient_dataframe = train_data[train_data['PATIENT_KEY'] == patient_key]
        testing_patient_dataframe = test_data[test_data['PATIENT_KEY'] == patient_key]

        for test_index, testing_patient in testing_patient_dataframe.iterrows():
            last_encounter_time = pd.datetime(1, 1, 1, 7, 0, 0)
            show_frequency = ''
            for _, training_patient in training_patient_dataframe.iterrows():
                testing_time = testing_patient['ENCOUNTER_APPOINTMENT_DATETIME']
                training_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
                if testing_time > training_time > last_encounter_time:
                    show_frequency = training_patient['SHOW_FREQUENCY']
                    last_encounter_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
            if show_frequency:
                test_data.loc[test_index, 'SHOW_FREQUENCY'] = show_frequency

    # storing the datset on files
    train_data.to_csv('datasets/training.csv', index=False, sep='|')
    test_data.to_csv('datasets/testing.csv', index=False, sep='|')

    # remove the PATIENT_ID,  ENCOUNTER_APPOINTMENT_DATETIME and NOSHOW columns
    train_data = train_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1)
    test_data = test_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1)

    logging.getLogger('tab.regular').debug('Finished calculating show frequency')

    return np.array(train_data), np.array(test_data)


def grid_search(dataset, y):

    logging.getLogger('regular').info('creating training and testing dataset')
    x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20, random_state=random_state)

    msg = 'adding the proportion or show_frequency column of how many times the patient has shown up to the ' \
          'appointment default = 1 i.e. it has a probability of showing up of 100\%'
    logging.getLogger('regular.time').debug(msg)
    x_train = x_train.assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(x_train)[0])))
    x_test = x_test.assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(x_test)[0])))

    # calculate patient's show_frequency
    x_train, x_test = calculate_show_frequency(train_data=x_train, test_data=x_test)

    # logging.getLogger('regular').info('normalizing training and test dataset')
    # x_train = np.array(x_train - x_train.mean() / (x_train.max() - x_train.min()))
    # x_test = np.array(x_test - x_test.mean() / (x_test.max() - x_test.min()))

    # create model
    logging.getLogger('regular.time').info('setting up model')

    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
    # batch_size = [10, 20, 40, 60, 80, 100]
    batch_size = [40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    # create object model and start training
    logging.getLogger('regular').debug('creating GridSearchCV object')
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=True)

    logging.getLogger('regular').info('training model')
    logging.getLogger('regular').debug('training dataset size = {0}'.format(np.shape(x_train)))
    logging.getLogger('regular').debug('testing dataset size = {0}'.format(np.shape(x_test)))
    grid_result = grid.fit(X=x_train, y=y_train)

    grid_score = grid_result.score(x_test, y_test)

    logging.getLogger('regular').info('grid score = {0}'.format(grid_score))

    logging.getLogger('regular').info('summarize results')
    logging.getLogger('regular').info('Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        logging.getLogger('regular').info('{0} ({1}) with: {3}'.format(mean, stdev, param))

    y_predictions = model.predict(x_test).astype(int).flatten()
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predictions, pos_label=2)
    print('AUC = {0}'.format(metrics.auc(fpr, tpr)))

    logging.getLogger('line.regular.time.line').info('finished running.')


def run_model(dataset, y):

    logging.getLogger('regular').info('creating training and testing dataset')
    x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33, random_state=42) 

    # adding the proportion or show_frequency column of how many times the patient has shown up to the
    # appointment default = 1 i.e. it has a probability of showing up of 100
    # creating SHOW_FREQUENCY column
    x_train = x_train.assign(SHOW_FREQUENCY=np.ones(np.shape(x_train)[0]))
    x_test = x_test.assign(SHOW_FREQUENCY=np.ones(np.shape(x_test)[0]))

    logging.getLogger('regular').info('calculating patient\'s show_frequency')
    x_train, x_test = calculate_show_frequency(train_data=x_train, test_data=x_test)

    logging.getLogger('regular').debug('creating and compiling model')
    model = Sequential()
    model.add(Dense(12, input_dim=np.shape(x_train)[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    logging.getLogger('regular').info('training model')
    logging.getLogger('regular').debug('training dataset size = {0}'.format(np.shape(x_train)))
    logging.getLogger('regular').debug('testing dataset size = {0}'.format(np.shape(x_test)))
    model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=1)

    logging.getLogger('regular').info('evaluating model')
    scores = model.evaluate(x_test, y_test, verbose=0)
    logging.getLogger('regular').info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def cross_validation(dataset, y):

    logging.getLogger('regular').info('define 10-fold cross validation test harness')
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_scores = list()
    for train, test in k_fold.split(dataset, y):
        msg = 'adding the proportion or show_frequency column of how many times the patient has shown up to the ' \
              'appointment default = 1 i.e. it has a probability of showing up of 100\%'
        logging.getLogger('regular.time').debug(msg)
        x_train = dataset.iloc[train].assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(dataset.iloc[train])[0])))
        x_test = dataset.iloc[test].assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(dataset.iloc[test])[0])))

        logging.getLogger('regular').info('calculate patient\'s show_frequency')
        x_train, x_test = calculate_show_frequency(train_data=x_train, test_data=x_test)

        logging.getLogger('regular').debug('creating and compiling model')
        model = Sequential()
        model.add(Dense(12, input_dim=np.shape(x_train)[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logging.getLogger('regular').info('training model')
        logging.getLogger('regular').debug('training dataset size = {0}'.format(np.shape(x_train)))
        logging.getLogger('regular').debug('testing dataset size = {0}'.format(np.shape(x_test)))
        model.fit(x_train, y[train], epochs=150, batch_size=10, verbose=1)

        logging.getLogger('regular').info('evaluating model')
        scores = model.evaluate(x_test, y[test], verbose=0)
        logging.getLogger('regular').info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)
    logging.getLogger('regular').info("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))


def main():

    # ignore warning of compiling tensorflow from source
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='dataset file', required=True)
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    parser.add_argument('-cv', '--cross_validation', action='store_true')
    parser.add_argument('-gs', '--grid_search', action='store_true')
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('line.regular.time.line').info('Running No_Show script')
    
    # import data from file
    logging.getLogger('regular').info('reading data from file')
    dataset = pd.read_csv(filepath_or_buffer=args.input_file, delimiter='|')

    # encode class values as integers
    encoder = LabelEncoder()
    categorical_keys = ['ENCOUNTER_DEPARTMENT_ABBR', 'ENCOUNTER_DEPARTMENT_SPECIALTY', 'ENCOUNTER_APPOINTMENT_WEEK_DAY',
                        'ENCOUNTER_APPOINTMENT_TYPE', 'PATIENT_GENDER']

    dataset_floats = dataset.copy()

    for key in categorical_keys:
        dataset_floats[key] = encoder.fit_transform(dataset[key])

    # remove every row that is missing a value
    dataset_floats.dropna(axis=0, inplace=True)

    # labels 0 == SHOWUP, 1 == NOSHOW
    y = np.array(dataset_floats['NOSHOW'])

    dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'] = pd.to_datetime(dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'])

    number_ones = len(y[y == 1])
    msg = 'data points NOSHOW true = {0}'.format(number_ones)
    logging.getLogger('regular').debug(msg)
    number_zeros = len(y[y == 0])
    msg = 'data points NOSHOW False = {0}'.format(number_zeros)
    logging.getLogger('regular').debug(msg)

    # check if cross validation flag is set
    if args.cross_validation:
        logging.getLogger('regular').info('running cross validation')
        cross_validation(dataset=dataset_floats, y=y)
    if args.grid_search:
        logging.getLogger('regular').info('running grid search')
        grid_search(dataset=dataset_floats, y=y)
    else:
        logging.getLogger('regular').info('running basic NN model')
        run_model(dataset=dataset_floats, y=y)


if __name__ == '__main__':
    main()
