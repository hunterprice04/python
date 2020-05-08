from sklearn import svm, metrics
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class SSM:
    def __init__(self):
        self.data = 0
        self.target = 0
        self.x_train = 0
        self.x_test = 0
        self.y_train = 0
        self.y_test = 0

    def read_csv(self, csv, sep=",", usecols=None):
        # here we are reading in the csv file and converting it to a pandas data frame
        data_frame = pd.read_csv(csv, sep=sep, usecols=usecols)

        # Split the data into categorical and quantitative data frames
        cat_data_frame = data_frame.select_dtypes(include=['object'])
        quant_data_frame = data_frame.select_dtypes(exclude=['object'])

        # encode the categorical data
        enc = OneHotEncoder()
        enc.fit(cat_data_frame)
        cat_data_frame = enc.transform(cat_data_frame).toarray()

        # here we are converting the pandas quantitative data frame into a numpy array
        quant_data_frame = pd.DataFrame(quant_data_frame).to_numpy()

        # here we are concatenating the categorical data back with the quantitative data
        # data_frame = np.concatenate((cat_data_frame, quant_data_frame), axis=1)
        data_frame = quant_data_frame
        print(data_frame)
        # here we are splitting the data frame into either the data we will use
        # or the target that we are looking to predict
        self.data = data_frame[:, :-1]
        self.target = data_frame[:, -1:].reshape(data_frame.shape[0])

    def split_data(self, test_size, random_state):
        # here we are splitting the data into batches that we will use to train the model
        # and batches we will ise
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def svc_linear(self):
        # parameter_candidates = [
        #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #
        # ]
        #
        # clf = GridSearchCV(estimator=svm.SVR(), param_grid=parameter_candidates, n_jobs=1)
        clf = svm.SVR(C=200, kernel='linear', gamma=0.001)

        print("C")
        clf.fit(self.x_train, self.y_train)
        print('Score: ', clf.score(self.x_test, self.y_test))
        predicted = clf.predict(self.x_test)
        print("Predicted Values:\n", predicted)
        print("Target Values:\n", self.y_test)

        # print("Classification Report:\n", metrics.classification_report(self.y_test, predicted))
        #
        # # Create a plot with subplots in a grid of 1X2
        # x_iso = Isomap().fit_transform(self.x_train)
        #
        # fig = plt.figure(1, (8, 4))
        # gs = gridspec.GridSpec(1, 2)
        # ax = [fig.add_subplot(ss) for ss in gs]
        #
        # # Adjust the layout
        # fig.subplots_adjust(top=0.85)
        #
        # # Add title
        # fig.suptitle('Predicted Versus Actual Labels', fontsize=14, fontweight='bold')
        #
        # # Add scatterplots to the subplots
        # ax[0].scatter(x_iso[:, 0], x_iso[:, 1], c=predicted, edgecolors='black')
        # ax[0].set_title('Predicted labels')
        # ax[1].scatter(x_iso[:, 0], x_iso[:, 1], c=self.y_train, edgecolors='black')
        # ax[1].set_title('Actual Labels')
        #
        # gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        # plt.show()
        # # Print out the results
        # print('Best score for training data:', clf.best_score_)
        # print('Best `C`:', clf.best_estimator_.C)
        # print('Best kernel:', clf.best_estimator_.kernel)
        # print('Best `gamma`:', clf.best_estimator_.gamma)
