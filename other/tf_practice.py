import os
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


def main():

    root_dir = os.path.abspath('../')
    data_dir = os.path.join(root_dir, 'data')

    # load pima indians dataset
    dataset = numpy.loadtxt(os.path.join(data_dir, 'pima-indians-diabetes.csv'), delimiter=",")
    # split into input (x) and output (y) variables
    x = dataset[:, 0:8]
    y = dataset[:, 8]

    import IPython
    IPython.embed()

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x, y, epochs=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(x, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    main()
