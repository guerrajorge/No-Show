import argparse
import logging
from utils.logger import logger_initialization
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split


def load(train_dataset=pd.DataFrame(), test_dataset=pd.DataFrame):
    global train_data
    global test_data

    if not train_dataset.empty:
        train_data = train_dataset
    if not test_dataset.empty:
        test_data = test_dataset


def store_dataset(training_dir='', testing_dir=''):

    logging.getLogger('regular').info('storing data')

    logging.getLogger('regular').debug('processed dataset information')
    logging.getLogger('regular').debug('training dataset shape (before storing)= {0}'.format(train_data.shape))
    logging.getLogger('regular').debug('training dataset keys (before storing) = {0}'.format(train_data.keys()))
    logging.getLogger('regular').debug('testing dataset shape (before storing) = {0}'.format(test_data.shape))
    logging.getLogger('regular').debug('testing dataset keys (before storing) = {0}'.format(test_data.keys()))

    train_data.to_csv(training_dir, index=False, sep='|')
    test_data.to_csv(testing_dir, index=False, sep='|')


def calculate_train_freq(patient_info):
    # id_value = patient_info['data']
    # encounter_status = patient_info['status']
    # pivot_column = patient_info['pivot_column']
    # freq_column = patient_info['freq_column']
    id_value = patient_info[0]
    encounter_status = patient_info[1]
    pivot_column = patient_info[2]
    freq_column = patient_info[3]

    patient_dataframe = train_data[train_data[pivot_column] == id_value]

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
        if data_point['ENCOUNTER_APPOINTMENT_STATUS'] == encounter_status:
            total_no_shows += 1.0
            prob = total_no_shows / encounters_processed
        else:
            total_shows += 1.0
            prob = 1 - (total_shows / encounters_processed)

        # update the NOSHOW_FREQUENCY for the specific patient's based on the index processed
        df.loc[index, freq_column] = prob

    return df


def calculate_test_freq(patient_info):

    # id_value = patient_info['data']
    # pivot_column = patient_info['pivot_column']
    # freq_column = patient_info['freq_column']
    id_value = patient_info[0]
    pivot_column = patient_info[1]
    freq_column = patient_info[2]

    training_patient_dataframe = train_data[train_data[pivot_column] == id_value]
    testing_patient_dataframe = test_data[test_data[pivot_column] == id_value]

    test_df = testing_patient_dataframe.copy(deep=True)

    # For every point in the testing dataset that matches the training dataset, loop for the
    # ENCOUNTER_APPOINTMENT_DATETIME of the training points that occurred before the ENCOUNTER_APPOINTMENT_DATETIME
    # in the testing data point. Grab and use the latest one to update the testing dataset
    for test_index, testing_patient in testing_patient_dataframe.iterrows():
        last_encounter_time = pd.datetime(1, 1, 1, 7, 0, 0)
        freq_val = ''
        for _, training_patient in training_patient_dataframe.iterrows():
            testing_time = testing_patient['ENCOUNTER_APPOINTMENT_DATETIME']
            training_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
            if testing_time > training_time > last_encounter_time:
                freq_val = training_patient[freq_column]
                last_encounter_time = training_patient['ENCOUNTER_APPOINTMENT_DATETIME']
        if freq_val != '':
            # update the NOSHOW_FREQUENCY to the one obtained from the training dataset
            test_df.loc[test_index, freq_column] = freq_val

    return test_df


def calculate_train_rep(patient_info):

    # id_value = patient_info['data']
    # encounter_status = patient_info['status']
    # pivot_column = patient_info['pivot_column']
    # freq_column = patient_info['freq_column']
    id_value = patient_info[0]
    encounter_status = patient_info[1]
    pivot_column = patient_info[2]
    freq_column = patient_info[3]

    patient_dataframe = train_data[train_data[pivot_column] == id_value]

    df = patient_dataframe.copy(deep=True)

    df = df.sort_values('ENCOUNTER_APPOINTMENT_DATETIME', axis=0)

    total_count = 0
    # loop through each encounter
    for index, data_point in df.iterrows():
        # if the patient did not show up
        if data_point['ENCOUNTER_APPOINTMENT_STATUS'] == encounter_status:
            total_count += 1
        else:
            total_count = 0

        df.loc[index, freq_column] = total_count

    return df


def calculate_frequencies():
    """
    This function will loop through all the unique ENCOUNTER_DEPARTMENT_ABBR in the training dataset and calculate the
    DEPARTMENT NOSHOW or CANCEL frequency at each new encounter. Then it will go through the testing dataset and it will
    set the latest DEPARTMENT_[NOSHOW|CANCELED]_FRQ of the training data point based on the ENCOUNTER_APPOINTMENT_DATETIME
    :return: a filled 'DEPARTMENT_[NOSHOW|CANCELED]_FRQ' column in the training and testing dataset.
    """

    # the unique_training_department_info variable will store three pieces of information
    #   1. the unique IDs of the departments
    #   2. the status i.e. NOSHOW or CANCEL
    #   3. the column where the information will be stored
    # This information will be access in the calculate_prob_department_[train|test] functions

    for postfix in ['PATIENT_KEY', 'ABBR', 'SPECIALTY']:

        if 'PATIENT' not in postfix:
            encounter_type = 'ENCOUNTER_DEPARTMENT_{0}'.format(postfix)
        else:
            encounter_type = postfix

        # get the unique IDs because that's going to be the pivot column
        unique_training_department_info = pd.DataFrame(train_data[encounter_type].unique(), columns=['data'])
        msg = 'obtaining frequencies for the NOSHOW and CANCEL status for the {0} information'.format(
            encounter_type)
        logging.getLogger('tab.regular').debug(msg)

        for status in ['NO SHOW', 'CANCELED']:

            frequency_column = list()
            if 'ABBR' in postfix:
                frequency_column.append('DEPARTMENT_{0}_FRQ'.format(status.replace(' ', '')))
                frequency_column.append('NUM_{0}_DEPARTMENT'.format(status.replace(' ', '')))
            elif 'SPECIALTY' in postfix:
                frequency_column.append('SPECIALTY_{0}_FRQ'.format(status.replace(' ', '')))
                frequency_column.append('NUM_{0}_SPECIALTY'.format(status.replace(' ', '')))
            elif 'PATIENT' in postfix:
                frequency_column.append('PATIENT_{0}_FRQ'.format(status.replace(' ', '')))
                frequency_column.append('NUM_{0}_PATIENT'.format(status.replace(' ', '')))

            for frq_column in frequency_column:

                msg = 'processing {0} training data'.format(frq_column)
                logging.getLogger('tab.regular.time').info(msg)

                # the resulting column data will be stored in this dataframe
                processed_train_data = pd.DataFrame()
                # other information passed to the function
                unique_training_department_info['status'] = status
                unique_training_department_info['pivot_column'] = encounter_type
                unique_training_department_info['freq_column'] = frq_column

                if 'FRQ' in frq_column:
                    # multi-processed
                    pool = Pool(processes=30)
                    processed_train_data = processed_train_data.append(pool.map(
                        calculate_train_freq, list(unique_training_department_info.values)), ignore_index=True)

                    # # sequential
                    # # department ID (did)
                    # for _, did in unique_training_department_info.iterrows():
                    #     processed_train_data = processed_train_data.append(calculate_train_freq(did))
                else:

                    # multi-processed
                    pool = Pool(processes=30)
                    processed_train_data = processed_train_data.append(pool.map(
                        calculate_train_rep, list(unique_training_department_info.values)), ignore_index=True)

                    # # sequential
                    # # department ID (did)
                    # for _, did in unique_training_department_info.iterrows():
                    #     processed_train_data = processed_train_data.append(calculate_train_rep(did))

                msg = 'finished processing {0} training data'.format(frq_column)
                logging.getLogger('tab.regular.time').info(msg)
                # update the training dataset
                load(train_dataset=processed_train_data)

                unique_testing_department_ids = test_data[encounter_type].unique()

                # obtaining the IDs that are in both the training and the testing dataset since those are the only ones
                # that need processing
                unique_testing_in_training = pd.unique(train_data[train_data[encounter_type].isin(
                    unique_testing_department_ids)][encounter_type])

                processed_test_data = pd.DataFrame()

                msg = 'processing {0} testing data'.format(frq_column)
                logging.getLogger('tab.regular.time').info(msg)
                # only go through the whole processing if there are common IDs in the training and testing dataset
                if len(unique_testing_in_training) != 0:
                    testing_info = pd.DataFrame(data=unique_testing_in_training, columns=['data'])
                    testing_info['pivot_column'] = encounter_type
                    testing_info['freq_column'] = frq_column

                    # multi-processed
                    pool = Pool(processes=30)
                    processed_test_data = processed_test_data.append(pool.map(
                        calculate_test_freq, list(testing_info.values)), ignore_index=True)

                    # # sequential
                    # # department ID = did
                    # for _, did in testing_info.iterrows():
                    #     processed_test_data = processed_test_data.append(calculate_test_freq(did))

                    # after the new testing information has been processed, we need to update the testing dataset
                    for _, processed_point in processed_test_data.iterrows():
                        # get the row index of the current specific processed test data
                        # get the index of the same department and those encounter that happen previously to the current
                        # testing encounter
                        s_index = test_data[(test_data[encounter_type] ==
                                             processed_point[encounter_type]) &
                                            (test_data['ENCOUNTER_APPOINTMENT_DATETIME'] ==
                                             processed_point['ENCOUNTER_APPOINTMENT_DATETIME'])].index.values[0]
                        # modify its DEPARTMENT_[NOSHOW|CANCEL]_FREQUENCY value
                        test_data.loc[s_index, frq_column] = processed_point[frq_column]

                    load(test_dataset=test_data)

                msg = 'finished processing {0} testing data'.format(frq_column)
                logging.getLogger('tab.regular.time').debug(msg)


def process_data(dataset):
    logging.getLogger('regular').info('processing data')

    # encode class values as integers
    encoder = LabelEncoder()
    categorical_keys = ['ENCOUNTER_DEPARTMENT_ABBR', 'ENCOUNTER_DEPARTMENT_SPECIALTY',
                        'ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'PATIENT_GENDER']

    dataset_floats = dataset.copy()

    for key in categorical_keys:
        dataset_floats[key] = encoder.fit_transform(dataset[key])

    # remove every row that is missing a value
    # dataset_floats.dropna(axis=0, inplace=True)

    dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'] = pd.to_datetime(dataset_floats['ENCOUNTER_APPOINTMENT_DATETIME'])

    logging.getLogger('regular').info('creating training and testing dataset')
    x_train, x_test = train_test_split(dataset_floats, test_size=0.33, random_state=42)

    # Make sure all the data are sorted based on the date of the encounter
    x_train = x_train.sort_values('ENCOUNTER_APPOINTMENT_DATETIME', axis=0)
    x_test = x_test.sort_values('ENCOUNTER_APPOINTMENT_DATETIME', axis=0)

    # setting up global variables
    load(train_dataset=x_train, test_dataset=x_test)

    # PATIENT_NOSHOW_FREQUENCY No-show Rate - Patient's historical canceled rate
    # create relevant columns
    x_train = train_data.assign(PATIENT_NOSHOW_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(PATIENT_NOSHOW_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # PATIENT_CANCELED_FREQUENCY No-show Rate - Patient's historical canceled rate
    # create relevant columns
    x_train = train_data.assign(PATIENT_CANCELED_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(PATIENT_CANCELED_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # DEPARTMENT_NOSHOW_FRQ No-show Rate - Patient's historical no-show rate by department
    x_train = train_data.assign(DEPARTMENT_NOSHOW_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(DEPARTMENT_NOSHOW_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)
    # calculate_department_noshow_frequency()

    # DEPARTMENT_CANCEL_FRQ Cancellation Rate - Patient's historical cancellation rate by department
    x_train = train_data.assign(DEPARTMENT_CANCELED_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(DEPARTMENT_CANCELED_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)
    # calculate_department_cancel_frequency()

    # SPECIALTY_NOSHOW_FREQ No-show Rate Patient's historical no-show rate by department specialty
    x_train = train_data.assign(SPECIALTY_NOSHOW_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(SPECIALTY_NOSHOW_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # SPECIALTY_CANCEL_FREQ Cancellation Rate - Patient's historical cancellation rate by specialty
    x_train = train_data.assign(SPECIALTY_CANCELED_FRQ=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(SPECIALTY_CANCELED_FRQ=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive No-shows By PATIENT_KEY- Number of most recent consecutive no-shows patient has accrued
    x_train = train_data.assign(NUM_NOSHOW_PATIENT=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_NOSHOW_PATIENT=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive No-shows By ENCOUNTER_DEPARTMENT_ABBR - Number of most recent consecutive no-shows patient
    # has accrued by department
    x_train = train_data.assign(NUM_NOSHOW_DEPARTMENT=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_NOSHOW_DEPARTMENT=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive No-shows By ENCOUNTER_DEPARTMENT_SPECIALTY - Number of most recent consecutive no-shows patient
    # has accrued by specialty
    x_train = train_data.assign(NUM_NOSHOW_SPECIALTY=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_NOSHOW_SPECIALTY=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive canceled By PATIENT_KEY- Number of most recent consecutive no-shows patient has accrued
    x_train = train_data.assign(NUM_CANCELED_PATIENT=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_CANCELED_PATIENT=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive No-shows By ENCOUNTER_DEPARTMENT_ABBR - Number of most recent consecutive canceled patient
    # has accrued by department
    x_train = train_data.assign(NUM_CANCELED_DEPARTMENT=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_CANCELED_DEPARTMENT=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    # Consecutive No-shows By ENCOUNTER_DEPARTMENT_SPECIALTY - Number of most recent consecutive canceled patient
    # has accrued by specialty
    x_train = train_data.assign(NUM_CANCELED_SPECIALTY=np.zeros(np.shape(train_data)[0]))
    x_test = test_data.assign(NUM_CANCELED_SPECIALTY=np.zeros(np.shape(test_data)[0]))
    # update and populate the column with the right values
    load(train_dataset=x_train, test_dataset=x_test)

    calculate_frequencies()

    # # New PATIENT_KEY to Department - Has patient visited department in past 24 months
    # x_train = x_train.assign(DEPARTMENT_NEW_PATIENT=np.ones(np.shape(x_train)[0]))
    # x_test = x_test.assign(DEPARTMENT_NEW_PATIENT=np.ones(np.shape(x_test)[0]))
    #
    # # Days Since PATIENT_KEY Last Appointment - Number of days since patient's previous appointment
    # x_train = x_train.assign(PATIENT_LAST_APPT=np.ones(np.shape(x_train)[0]))
    # x_test = x_test.assign(PATIENT_LAST_APPT=np.ones(np.shape(x_test)[0]))

    logging.getLogger('regular').debug('training dataset shape = {0}'.format(x_train.shape))
    logging.getLogger('regular').debug('training dataset keys = {0}'.format(x_train.keys()))
    logging.getLogger('regular').debug('testing dataset shape = {0}'.format(x_test.shape))
    logging.getLogger('regular').debug('testing dataset keys = {0}'.format(x_test.keys()))


def load_data(input_file):

    logging.getLogger('regular').info('reading data from file')
    dataset = pd.read_csv(filepath_or_buffer=input_file, delimiter='|')

    logging.getLogger('regular').debug('dataset shape = {0}'.format(dataset.shape))
    logging.getLogger('regular').debug('dataset keys = {0}'.format(dataset.keys()))

    number_ones = len(dataset[dataset['NOSHOW'] == 1])
    msg = 'data points NOSHOW true = {0}'.format(number_ones)
    logging.getLogger('regular').debug(msg)
    number_zeros = len(dataset[dataset['NOSHOW'] == 0])
    msg = 'data points NOSHOW False = {0}'.format(number_zeros)
    logging.getLogger('regular').debug(msg)

    return dataset


def main():

    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='dataset file that has not being processed')
    parser.add_argument('-tr', '--train_file', help='processed training dataset file')
    parser.add_argument('-te', '--test_file', help='processed testing dataset file')
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    # import data from file
    dataset = load_data(args.input_file)
    # calculate relevant variables' values
    process_data(dataset=dataset)

    training_dir = args.train_file
    testing_dir = args.test_file

    # if those variables are not pass, then populate them
    if not (args.train_file and args.test_file):
        training_dir = 'datasets/train_data_processed.csv'
        testing_dir = 'datasets/test_data_processed.csv'

    # save it
    store_dataset(training_dir=training_dir, testing_dir=testing_dir)


if __name__ == '__main__':
    main()
