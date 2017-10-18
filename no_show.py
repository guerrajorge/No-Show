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

# seed for numpy and sklearn
random_state = 7
np.random.seed(random_state)


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def calculate_show_frequency(train_data, test_data):

    logging.getLogger('line.tab.regular').debug('Calculating show frequency')

    logging.getLogger('line.tab.regular').debug('obtain frequency of "show" and "noshow" based on the patient\'s id')
    noshow_df = train_data.groupby(['PATIENT_ID', 'NOSHOW']).size().reset_index(name='NOSHOW_FREQUENCY')
    noshow_df['NOSHOW_FREQUENCY'] = noshow_df['NOSHOW_FREQUENCY'].astype(float)

    logging.getLogger('line.tab.regular').debug('obtain the number of times a patient\'s record appears')
    patient_df = train_data.groupby(['PATIENT_ID']).size().reset_index(name='PATIENT_FREQUENCY')
    # convert the patient's record frequency to a dictionary in order to insert it on the dataframe
    patient_dict = patient_df.set_index('PATIENT_ID')['PATIENT_FREQUENCY'].to_dict()

    # insert the dictionary on the dataframe based on the PATIENT_ID and dictionaries' keys
    noshow_df['PATIENT_FREQUENCY'] = noshow_df['PATIENT_ID'].map(patient_dict)
    # change the type of the frequencies to float in order to divide them correctly
    noshow_df[['NOSHOW_FREQUENCY', 'PATIENT_FREQUENCY']] = \
        noshow_df[['NOSHOW_FREQUENCY', 'PATIENT_FREQUENCY']].astype(float)
    logging.getLogger('obtain show and no-show proportions')
    noshow_df['PATIENT_NOSHOW_FREQ'] = noshow_df['NOSHOW_FREQUENCY'].div(noshow_df['PATIENT_FREQUENCY'], axis=0)

    # only keep the probabilities of a patient showing up to the encounter and their PATIENT_ID
    present_patients_probabilities = noshow_df.loc[noshow_df['NOSHOW'] == 0][['PATIENT_ID', 'PATIENT_NOSHOW_FREQ']]
    present_patients_prob_dict = present_patients_probabilities.set_index('PATIENT_ID')['PATIENT_NOSHOW_FREQ'].to_dict()

    # update train and test data information
    train_data['SHOW_FREQUENCY'] = train_data['PATIENT_ID'].map(present_patients_prob_dict)
    test_data['SHOW_FREQUENCY'] = test_data['PATIENT_ID'].map(present_patients_prob_dict)

    # if the patient has not being seeind in the training dataset, assume the patient will show-up: replace NaN with 1
    test_data['SHOW_FREQUENCY'].fillna(value=1.0, inplace=True)
    train_data['SHOW_FREQUENCY'].fillna(value=1.0, inplace=True)

    # remove the PATIENT_ID and NOSHOW columns
    train_data = train_data.drop(['PATIENT_ID', 'NOSHOW'], axis=1)
    test_data = test_data.drop(['PATIENT_ID', 'NOSHOW'], axis=1)

    logging.getLogger('line.tab.regular').debug('Finished calculating show frequency')

    return np.array(train_data), np.array(test_data)


def grid_search(x_train, y_train, x_test, y_test):

    # adding the proportion or show_frequency column of how many times the patient has shown up to the appointment,
    # default = 1 i.e. it has a probability of showing up of 100%
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
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    # create object model and start training
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=True)

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

    # calculate patient's show_frequency
    x_train, x_test = calculate_show_frequency(train_data=x_train, test_data=x_test)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=np.shape(x_train)[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=1)
    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    logging.getLogger('regular').info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def cross_validation(dataset, y):

    # define 10-fold cross validation test harness
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_scores = list()
    for train, test in k_fold.split(dataset, y):
        # adding the proportion or show_frequency column of how many times the patient has shown up to the appointment,
        # default = 1 i.e. it has a probability of showing up of 100%
        x_train = dataset.iloc[train].assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(dataset.iloc[train])[0])))
        x_test = dataset.iloc[test].assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(dataset.iloc[test])[0])))

        # calculate patient's show_frequency
        x_train, x_test = calculate_show_frequency(train_data=x_train, test_data=x_test)

        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=np.shape(x_train)[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(x_train, y[train], epochs=150, batch_size=10, verbose=1)
        # evaluate the model
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
    o_dataset = pd.read_csv(filepath_or_buffer=args.input_file, delimiter='|')

    noshow_1 = o_dataset[o_dataset['NOSHOW'] == 1]
    noshow_0 = o_dataset[o_dataset['NOSHOW'] == 0]
    noshow_1.reset_index(drop=True, inplace=True)
    noshow_0.reset_index(drop=True, inplace=True)
    dataset = noshow_1.copy()
    dataset = dataset.append(noshow_0[:noshow_1.shape[0]]).copy()

    # labels 0 == SHOWUP, 1 == NOSHOW
    y = np.array(dataset['NOSHOW'])

    dataset.drop(['ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'ENCOUNTER_CLASS',
                  'ENCOUNTER_CLASS', 'ENCOUNTER_DEPARTMENT_SPECIALTY', 'NENCOUNTERTYPE', 'ENCOUNTER_PATIENT_AGE'],
                 inplace=True, axis=1)

    # # encode class values as integers
    # encoder = LabelEncoder()
    # categorical_keys = ['ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'ENCOUNTER_CLASS',
    #                     'ENCOUNTER_DEPARTMENT_SPECIALTY']
    # for key in categorical_keys:
    #     dataset[key] = encoder.fit_transform(dataset[key])
    #

    # check if cross validation flag is set
    if args.cross_validation:
        cross_validation(dataset=dataset, y=y)
    if args.grid_search:
        # logging.getLogger('regular').info('creating training and testing dataset')
        x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20, random_state=random_state)
        grid_search(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    else:
        run_model(dataset=dataset, y=y)


if __name__ == '__main__':
    main()
