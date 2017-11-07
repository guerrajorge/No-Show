import argparse
import pandas as pd
import os
import numpy as np
import logging
from utils.logger import logger_initialization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from multiprocessing import Pool, Value, Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# seed for numpy and sklearn
random_state = 1
np.random.seed(random_state)


def load(train_dataset=pd.DataFrame(), test_dataset=pd.DataFrame):
    global train_data
    global test_data

    if not train_dataset.empty:
        train_data = train_dataset
    if not test_dataset.empty:
        test_data = test_dataset


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


def calculate_prob_encounter(patient_info):
    patient_id = patient_info[0]
    patient_index = patient_info[1]

    patient_dataframe = train_data[train_data['PATIENT_KEY'] == patient_id]

    unique_training_patient_ids = train_data['PATIENT_KEY'].unique()

    msg = 'processing {0} out of {1} patients'.format(patient_index, len(unique_training_patient_ids) - 1)
    logging.getLogger('tab.tab.regular').debug(msg)

    df = patient_dataframe.copy(deep=True)
    # number of encounters processed
    encounters_processed = 0.0
    # total number of encounter that the patient showed up
    total_shows = 0.0
    # total number of encounter that the patient did not show up
    total_no_shows = 0.0
    # loop through each encounter
    for index, data_point in df.iterrows():
        encounters_processed += 1.0
        # if the patient did not show up
        if data_point['NOSHOW']:
            total_no_shows += 1.0
            prob = 1 - (total_no_shows / encounters_processed)
        else:
            total_shows += 1.0
            prob = total_shows / encounters_processed

        # update the SHOW_FREQUENCY for the specific patient's based on the index processed
        df.loc[index, 'SHOW_FREQUENCY'] = prob

    return df


def calculate_prob_encounter_test(patient_id):

    training_patient_dataframe = train_data[train_data['PATIENT_KEY'] == patient_id]
    testing_patient_dataframe = test_data[test_data['PATIENT_KEY'] == patient_id]

    test_df = testing_patient_dataframe.copy(deep=True)

    # For every point in the testing dataset that matches the training dataset, loop for the
    # ENCOUNTER_APPOINTMENT_DATETIME of the training points that occurred before the ENCOUNTER_APPOINTMENT_DATETIME
    # in the testing data point. Grab and use the latest one to update the testing dataset
    for test_index, testing_patient in testing_patient_dataframe.iterrows():
        last_encounter_time = pd.datetime(1, 1, 1, 7, 0, 0)
        show_frequency = ''
        for _, training_patient in training_patient_dataframe.iterrows():
            testing_time = testing_patient['ENCOUNTER_APPOINTMENT_DATETIME']
            training_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
            if testing_time > training_time > last_encounter_time:
                show_frequency = training_patient['SHOW_FREQUENCY']
                last_encounter_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
        if show_frequency != '':
            # update the SHOW_FREQUENCY to the one obtained from the training dataset
            test_df.loc[test_index, 'SHOW_FREQUENCY'] = show_frequency

    return test_df


def calculate_show_frequency(store_results=False):
    """
    This function will loop through all the unique PATIENT_IDs in the training dataset and calculate the SHOW frequency
    at each new encounter. Then it will go through the testing dataset and it will set the latest SHOW_FREQUENCY of the
    training data point based on the ENCOUNTER_APPOINTMENT_DATETIME
    :return: a filled 'SHOW_FREQUENCY' column in the training and testing dataset.
    """

    logging.getLogger('tab.regular').debug('obtaining frequency of "SHOW" and "NOSHOW" based on the patient\'s id')

    unique_training_patient_ids = pd.DataFrame(train_data['PATIENT_KEY'].unique())
    msg = 'there are {0} unique patient IDs in the training dataset.'.format(len(unique_training_patient_ids))
    logging.getLogger('tab.regular').debug(msg)
    logging.getLogger('tab.regular').debug('The first 5 patients, in the training dataset, are: {0}'.format(
        unique_training_patient_ids[:5].values))

    processed_train_data = pd.DataFrame()
    # index are used for debugging when running the script
    unique_training_patient_ids['index'] = unique_training_patient_ids.index

    logging.getLogger('tab.regular').info('processing training data')
    pool = Pool(processes=30)
    processed_train_data = processed_train_data.append(pool.map(calculate_prob_encounter,
                                                                list(unique_training_patient_ids.values)),
                                                       ignore_index=True)

    # update the training dataset
    load(train_dataset=processed_train_data)

    unique_testing_patient_ids = test_data['PATIENT_KEY'].unique()
    msg = 'there are {0} unique patient IDs in the training dataset.'.format(len(unique_testing_patient_ids))
    logging.getLogger('line.tab.regular').debug(msg)
    logging.getLogger('tab.regular').debug('The first 5 patients, in the testing dataset, are: {0}'.format(
        unique_testing_patient_ids[:5]))

    unique_testing_in_training = pd.unique(train_data[train_data['PATIENT_KEY'].isin(
        unique_testing_patient_ids)]['PATIENT_KEY'])

    processed_test_data = pd.DataFrame()

    logging.getLogger('tab.regular.time').info('processing testing data')
    if len(unique_testing_in_training) != 0:
        # multi-processed
        pool = Pool(processes=20)
        processed_test_data = processed_test_data.append(pool.map(calculate_prob_encounter_test,
                                                                  unique_testing_in_training),
                                                         ignore_index=True)
        # sequential
        # for uid in unique_testing_in_training:
        #     processed_test_data = processed_test_data.append(calculate_prob_encounter_test(uid))

        for _, processed_point in processed_test_data.iterrows():
            s_index = test_data[(test_data['PATIENT_KEY'] == processed_point['PATIENT_KEY']) & \
                                (test_data['ENCOUNTER_APPOINTMENT_DATETIME'] ==
                                 processed_point['ENCOUNTER_APPOINTMENT_DATETIME'])].index.values[0]
            test_data.loc[s_index, 'SHOW_FREQUENCY'] = processed_point['SHOW_FREQUENCY']

        load(test_dataset=test_data)

    if store_results:

        logging.getLogger('regular').debug('processed dataset information')
        logging.getLogger('regular').debug('training dataset shape (before storing)= {0}'.format(train_data.shape))
        logging.getLogger('regular').debug('training dataset keys (before storing) = {0}'.format(train_data.keys()))
        logging.getLogger('regular').debug('testing dataset shape (before storing) = {0}'.format(test_data.shape))
        logging.getLogger('regular').debug('testing dataset keys (before storing) = {0}'.format(test_data.keys()))

        train_data.to_csv('datasets/train_data_processed.csv', index=False, sep='|')
        test_data.to_csv('datasets/test_data_processed.csv', index=False, sep='|')

        # remove the NOSHOW columns
        # remove the PATIENT_ID,  ENCOUNTER_APPOINTMENT_DATETIME and NOSHOW columns
        load(train_dataset=train_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1),
             test_dataset=test_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1))

    else:
        # remove the PATIENT_ID,  ENCOUNTER_APPOINTMENT_DATETIME and NOSHOW columns
        load(train_dataset=train_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1),
             test_dataset=processed_test_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'NOSHOW'], axis=1))

    logging.getLogger('tab.regular.time').debug('Finished calculating show frequency')

    return np.array(train_data), np.array(test_data)


def run_model(dataset='', y='', pre_process=True, training_data='', testing_data='', training_y='', testing_y='',
              store_db=False):

    if not pre_process:
        logging.getLogger('regular').info('creating training and testing dataset')
        x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33, random_state=42)

        # adding the proportion or show_frequency column of how many times the patient has shown up to the
        # appointment default = 1 i.e. it has a probability of showing up of 100
        # creating SHOW_FREQUENCY column
        x_train = x_train.assign(SHOW_FREQUENCY=np.ones(np.shape(x_train)[0]))
        x_test = x_test.assign(SHOW_FREQUENCY=np.ones(np.shape(x_test)[0]))

        logging.getLogger('regular').debug('training dataset shape = {0}'.format(x_train.shape))
        logging.getLogger('regular').debug('training dataset keys = {0}'.format(x_train.keys()))
        logging.getLogger('regular').debug('testing dataset shape = {0}'.format(x_test.shape))
        logging.getLogger('regular').debug('testing dataset keys = {0}'.format(x_test.keys()))

        load(train_dataset=x_train, test_dataset=x_test)

        logging.getLogger('regular.time').info('calculating patient\'s show_frequency')
        x_train, x_test = calculate_show_frequency(store_results=store_db)

        logging.getLogger('regular').debug('training dataset processed shape = {0}'.format(x_train.shape))
        logging.getLogger('regular').debug('testing dataset processed shape = {0}'.format(x_test.shape))

    else:
        x_train = training_data
        x_test = testing_data
        y_train = training_y
        y_test = testing_y

    logging.getLogger('regular.time').debug('creating and compiling model')
    model = Sequential()
    model.add(Dense(12, input_dim=np.shape(x_train)[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    logging.getLogger('regular.time').info('training model')
    logging.getLogger('regular').debug('training dataset size processed = {0}'.format(np.shape(x_train)))
    logging.getLogger('regular').debug('testing dataset size processed = {0}'.format(np.shape(x_test)))
    model.fit(x_train, y_train, epochs=150, batch_size=5, verbose=1)

    logging.getLogger('regular.time').info('evaluating model')
    scores = model.evaluate(x_test, y_test, verbose=0)
    logging.getLogger('regular').info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def main():
    # ignore warning of compiling tensorflow from source
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='dataset file that has not being processed')
    parser.add_argument('-tr', '--train_file', help='processed training dataset file')
    parser.add_argument('-te', '--test_file', help='processed testing dataset file')
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    parser.add_argument('-cv', '--cross_validation', action='store_true')
    parser.add_argument('-gs', '--grid_search', action='store_true')
    parser.add_argument('-p', '--processed_dataset', action='store_true', help='this flag is used when the training '
                                                                               'and testing datasets are provided')
    parser.add_argument('-s', '--store_datasets', action='store_true', help='this flag is used to store the training'
                                                                            'and testing dataset on local system')
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('line.regular.time.line').info('Running No_Show script')

    # import data from file
    logging.getLogger('regular').info('reading data from file')

    # initializing the variables
    dataset_floats = ''
    y = ''
    x_train_data = ''
    x_test_data = ''
    y_train_data = ''
    y_test_data = ''

    if not args.processed_dataset:
        dataset = pd.read_csv(filepath_or_buffer=args.input_file, delimiter='|')

        # encode class values as integers
        encoder = LabelEncoder()
        categorical_keys = ['ENCOUNTER_DEPARTMENT_ABBR', 'ENCOUNTER_DEPARTMENT_SPECIALTY',
                            'ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'PATIENT_GENDER']

        dataset_floats = dataset.copy()

        for key in categorical_keys:
            dataset_floats[key] = encoder.fit_transform(dataset[key])

        # remove every row that is missing a value
        dataset_floats.dropna(axis=0, inplace=True)

        # labels 0 == SHOWUP, 1 == NOSHOW
        y = np.array(dataset_floats['NOSHOW'])

        dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'] = pd.to_datetime(
            dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'])

        logging.getLogger('regular').debug('dataset shape = {0}'.format(dataset_floats.shape))
        logging.getLogger('regular').debug('dataset keys = {0}'.format(dataset_floats.keys()))

        number_ones = len(y[y == 1])
        msg = 'data points NOSHOW true = {0}'.format(number_ones)
        logging.getLogger('regular').debug(msg)
        number_zeros = len(y[y == 0])
        msg = 'data points NOSHOW False = {0}'.format(number_zeros)
        logging.getLogger('regular').debug(msg)

    else:

        tr_data = pd.read_csv(filepath_or_buffer=args.train_file)
        te_data = pd.read_csv(filepath_or_buffer=args.test_file)

        logging.getLogger('regular').debug('training dataset shape = {0}'.format(tr_data.shape))
        logging.getLogger('regular').debug('training dataset keys = {0}'.format(tr_data.keys()))
        logging.getLogger('regular').debug('testing dataset shape = {0}'.format(te_data.shape))
        logging.getLogger('regular').debug('testing dataset keys = {0}'.format(te_data.keys()))

        y_train_data = tr_data['NOSHOW'].values
        y_test_data = te_data['NOSHOW'].values
        x_train_data = tr_data.drop('NOSHOW', axis=1).values
        x_test_data = te_data.drop('NOSHOW', axis=1).values

    # check if cross validation flag is set
    logging.getLogger('regular').info('running basic NN model')
    run_model(dataset=dataset_floats, y=y, training_data=x_train_data, testing_data=x_test_data,
              training_y=y_train_data, testing_y=y_test_data, pre_process=args.processed_dataset,
              store_db=args.store_datasets)


if __name__ == '__main__':
    main()
