import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from utils.logger import logger_initialization


np.random.seed(7)


def calculate_noshow_frequency(train_data, test_data):

    # keep the patient's information
    #   key = patient id
    #   value = (list) [# of show appointments, # of appointments]
    patient_record = dict()

    logging.getLogger('line.tab.regular').debug('Calculating history (training dataset)')

    # loop trough all the IDs and find their frequency then calculate the average
    for current_id in train_data['PATIENT_ID'].unique():

        logging.getLogger('tab.regular').debug('patient id = {0}'.format(current_id))

        # find the other encounter of the current patient
        patient_encounters = train_data.index[train_data['PATIENT_ID'] == current_id]
        total_patient_encounters = float(len(patient_encounters))

        logging.getLogger('tab.regular').debug('number of records = {0}'.format(total_patient_encounters))

        # calculate the average of the number of time the patient show up to an appointment
        total_sum = train_data.loc[patient_encounters]['NOSHOW'].sum()
        average_show_count = total_sum / total_patient_encounters

        logging.getLogger('tab.regular.line').debug('average "no show" frequency = {0}'.format(average_show_count))

        # insert the average for all the instances of that patient
        train_data.set_value(patient_encounters, 'NOSHOW_FREQUENCY', average_show_count)

        patient_record[current_id] = average_show_count

    logging.getLogger('tab.regular').debug('Calculating history (testing dataset)')

    # loop through all the IDs in the test dataset, check if that ID was present in the
    # training dataset, modify its frequency (if necessary), calculate the frequency's average
    # and insert it in its history
    for current_id in test_data['PATIENT_ID'].unique():

        # obtaining the indices of that current patient_id in the test dataset
        patient_encounters = test_data.index[test_data['PATIENT_ID'] == current_id]

        logging.getLogger('tab.regular').debug('patient id = {0}'.format(current_id))

        # check if ID in the training dataset
        if current_id in patient_record.keys():
            # obtain the training dataset values for that patient
            average_show_count = patient_record[current_id]
            logging.getLogger('tab.regular.line').debug('previous average "no show" frequency = {0}'.format(
                average_show_count))

            test_data.set_value(patient_encounters, 'NOSHOW_FREQUENCY', average_show_count)

    # remove the PATIENT_ID and NOSHOW columns
    train_data = train_data.drop(['PATIENT_ID', 'NOSHOW'], axis=1)
    test_data = test_data.drop(['PATIENT_ID', 'NOSHOW'], axis=1)

    return train_data, test_data


def main():

    # ignore warning about installing tensorflow from source
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='dataset file', required=True)
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'], type=str.upper,
                        help="Set the logging level")
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('line.regular.time.line').info('Running No_Show script')
    
    # import data from file
    logging.getLogger('regular').info('reading data from file')
    dataset = pd.read_csv(filepath_or_buffer=args.input_file, delimiter='|')

    # labels 0 == SHOWUP, 1 == NOSHOW
    y = np.array(dataset['NOSHOW'])

    # encode class values as integers
    encoder = LabelEncoder()
    categorical_keys = ['ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'ENCOUNTER_CLASS',
                        'ENCOUNTER_DEPARTMENT_SPECIALTY']
    for key in categorical_keys:
        dataset[key] = encoder.fit_transform(dataset[key])

    logging.getLogger('regular').info('creating training and testing dataset')
    x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33, random_state=42)

    x_train.reset_index(inplace=True)
    x_test.reset_index(inplace=True)

    x_train = x_train.assign(NOSHOW_FREQUENCY=pd.Series(np.zeros(np.shape(x_train)[0])))
    x_test = x_test.assign(NOSHOW_FREQUENCY=pd.Series(np.zeros(np.shape(x_test)[0])))

    # calcualte patient's history
    x_train, x_test = calculate_noshow_frequency(train_data=x_train, test_data=x_test)

    logging.getLogger('regular').info('normalizing training and test dataset')
    x_train = np.array(x_train - x_train.mean() / (x_train.max() - x_train.min()))
    x_test = np.array(x_test - x_test.mean() / (x_test.max() - x_test.min()))

    # create model
    logging.getLogger('regular').info('setting up model')
    model = Sequential()
    model.add(Dense(40, input_dim=np.shape(x_train)[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    logging.getLogger('regular').info('training model')
    model.fit(np.array(x_train), y_train, epochs=150, batch_size=10)

    # evaluate the model
    logging.getLogger('regular').info('evaluation model')
    scores = model.evaluate(np.array(x_test), y_test)
    logging.getLogger('regular').info("\nAccuracy = {0:.2f}".format(scores[1] * 100))

    # y_predictions = model.predict(x_test).astype(int).flatten()
    # fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predictions, pos_label=2)
    # print('AUC = {0}'.format(metrics.auc(fpr, tpr)))

    logging.getLogger('line.regular.time.line').info('finished running.')


if __name__ == '__main__':
    main()
