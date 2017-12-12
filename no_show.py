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
from sklearn import svm

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


def run_model(training_data='', testing_data='', training_y='', testing_y='', svm_flag=False, gs_flag=False):

    x_train = training_data
    x_test = testing_data
    y_train = training_y
    y_test = testing_y

    if svm_flag:

        if gs_flag:
        
            logging.getLogger('regular.time').info('running GRIDSEARCH SVM model')
            param_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
            model = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=-1)
            model.fit(x_train, y_train)
            logging.getLogger('regular.time').debug('finished training model')

            # View the accuracy score
            logging.getLogger('regular').debug('Best score for data1: {0}'.format(model.best_score_))

            # View the best parameters for the model found using grid search
            logging.getLogger('regular').debug('Best C: {0}'.format(model.best_estimator_.C))
            logging.getLogger('regular').debug('Best Kernel: {0}'.format(model.best_estimator_.kernel))
            logging.getLogger('regular').debug('Best Gamma: {0}'.format(model.best_estimator_.gamma))

        else:
            logging.getLogger('regular.time').info('running SVM model')
            model = svm.SVC()
            model.fit(x_train, y_train)
            logging.getLogger('regular.time').debug('finished training model')

        svm_score = model.score(x_test, y_test)
        logging.getLogger('regular').info("score: {0}".format(svm_score))

    else:

        logging.getLogger('regular').info('running basic NN model')
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
    parser.add_argument('-svm', '--svm', help='run support vector machine', action='store_true')
    parser.add_argument('-p', '--processed_dataset', action='store_true', help='this flag is used when the training '
                                                                               'and testing datasets are provided')
    parser.add_argument('-s', '--store_datasets', action='store_true', help='this flag is used to store the training'
                                                                            'and testing dataset on local system')
    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('line.regular.time.line').info('Running No_Show script')

    # import data from file
    logging.getLogger('regular').info('reading data from file')

    tr_data = pd.read_csv(filepath_or_buffer=args.train_file, delimiter='|')
    te_data = pd.read_csv(filepath_or_buffer=args.test_file, delimiter='|')

    logging.getLogger('regular').debug('training dataset shape = {0}'.format(tr_data.shape))
    logging.getLogger('regular').debug('training dataset keys = {0}'.format(tr_data.keys()))
    logging.getLogger('regular').debug('testing dataset shape = {0}'.format(te_data.shape))
    logging.getLogger('regular').debug('testing dataset keys = {0}'.format(te_data.keys()))

    y_train_data = tr_data['NOSHOW'].values
    y_test_data = te_data['NOSHOW'].values
    x_train_data = tr_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'ENCOUNTER_APPOINTMENT_STATUS',
                                 'NOSHOW'], axis=1).values
    x_test_data = te_data.drop(['PATIENT_KEY', 'ENCOUNTER_APPOINTMENT_DATETIME', 'ENCOUNTER_APPOINTMENT_STATUS',
                                'NOSHOW'], axis=1).values

    # check if cross validation flag is set
    run_model(training_data=x_train_data, testing_data=x_test_data, training_y=y_train_data, testing_y=y_test_data,
              svm_flag=args.svm, gs_flag=args.grid_search)


if __name__ == '__main__':
    main()
