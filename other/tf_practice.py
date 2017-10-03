# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import os


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main_1():
    # seed for numpy and sklearn
    random_state = 7
    numpy.random.seed(random_state)

    # ignore warning of compiling tensorflow from source
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    root_dir = os.path.abspath('../')
    data_dir = os.path.join(root_dir, 'data')

    # load pima indians dataset
    dataset = numpy.loadtxt(os.path.join(data_dir, 'pima-indians-diabetes.csv'), delimiter=",")
    # split into input (x) and output (y) variables
    x = dataset[:, 0:8]
    y = dataset[:, 8]

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def main_2():
    # seed for numpy and sklearn
    random_state = 7
    np.random.seed(random_state)

    # ignore warning of compiling tensorflow from source
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

    noshow_1 = dataset[dataset['NOSHOW'] == 1]
    noshow_0 = dataset[dataset['NOSHOW'] == 0]
    noshow_1.reset_index(drop=True, inplace=True)
    noshow_0.reset_index(drop=True, inplace=True)
    n_dataset = noshow_1.copy()
    n_dataset = n_dataset.append(noshow_0[:noshow_1.shape[0]]).copy()
    y = np.array(n_dataset['NOSHOW'])

    n_dataset.drop(['ENCOUNTER_APPOINTMENT_WEEK_DAY', 'ENCOUNTER_APPOINTMENT_TYPE', 'ENCOUNTER_CLASS',
                    'ENCOUNTER_CLASS', 'ENCOUNTER_DEPARTMENT_SPECIALTY', 'NENCOUNTERTYPE', 'ENCOUNTER_PATIENT_AGE'],
                     inplace=True, axis=1)


    x_train, x_test, y_train, y_test = train_test_split(n_dataset, y, test_size=0.20, random_state=random_state)

    x_train = x_train.assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(x_train)[0])))
    x_test = x_test.assign(SHOW_FREQUENCY=pd.Series(np.ones(np.shape(x_test)[0])))

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
    model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=2)

    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    main_1()
