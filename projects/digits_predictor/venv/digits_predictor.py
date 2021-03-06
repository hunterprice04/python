from sklearn import datasets, cluster, metrics, svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import Isomap
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DigitsPredictor:
    def __init__(self, digits):
        # digits is the main data set the we are working with
        self.digits = digits

        # data holds the data we will be training the model with
        self.data = digits.data

        # target holds the correct answer corresponding to the same index in data
        self.target = digits.target

        # images holds each index of data formatted as an 8x8 pixel array instead of a 64 pixel array
        self.images = digits.images

        # unique holds each unique target value
        self.unique = np.unique(digits.target)

    def display_data(self):
        # creates an image that is 6 inches by 6 inches big
        fig = plt.figure(figsize=(6, 6))

        # this configures the subplots
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        # will draw 64 of the images
        for i in range(64):
            # initializes each subplot on a 8x8 grid at the index i + 1
            ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

            # displays the image at the ith position in the data set
            ax.imshow(self.images[i], cmap=plt.cm.binary, interpolation='nearest')

            # prints out the images corresponding target value
            ax.text(0, 7, str(self.target[i]))

        # show the plot
        plt.show()

        # Join the images and target labels in a list
        images_and_labels = list(zip(self.images, self.target))

        # for every element in the list
        for i, (image, label) in enumerate(images_and_labels[:8]):
            # initialize a subplot of 2X4 at the i+1-th position
            plt.subplot(2, 4, i + 1)
            # Don't plot any axes
            plt.axis('off')
            # Display images in all subplots
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            # Add a title to each subplot
            plt.title('Training: ' + str(label))

        # Show the plot
        plt.show()

    def create_reduced_pca(self):
        # create a regular pca model
        pca = PCA(n_components=2)

        # fit and transform the data to the model
        reduced_data_pca = pca.fit_transform(self.data)

        # inspect the shape of the model
        print("Shape of PCA model:\n",reduced_data_pca.shape)

        # print out the data
        print("PCA data:\n",reduced_data_pca)

        # colors in the graph that will correspond to the target values
        colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

        # loop through each target value
        for i in range(len(colors)):
            # get each x and y of the pca model that has the current target value
            x = reduced_data_pca[:, 0][digits.target == i]
            y = reduced_data_pca[:, 1][digits.target == i]

            # plot the points onto the scatter plot
            plt.scatter(x, y, c=colors[i], edgecolors='black')

        # create the axis and graph titles and the legend
        plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title("PCA Scatter Plot")
        plt.show()

    def kmeans_model(self, test_size, random_state,show=None):
        # pre-process the data
        standardized_data = scale(self.data)

        # splitting the data into training and testing sets
        # typically 3/4 of the data is used to train, 1/4 of the data is used to test
        # x is the data you are testing : y is the target values of the corresponding data
        x_train, x_test, y_train, y_test, images_train, images_test = train_test_split(standardized_data, self.target,
                                                                                       self.images,
                                                                                       test_size=test_size,
                                                                                       random_state=random_state)
        # gets the number of training features
        n_samples, n_features = x_train.shape

        # print out the number of samples and features
        print("# of training samples: ", n_samples)
        print("# of training features: ", n_features)

        # num_digits is the amount of unique targets
        n_digits = len(np.unique(y_train))

        # create the KMeans model.
        # init defaults to init='k-means++'
        # add n-init argument to determine how many different centroid configurations the algorithm will try
        clf = cluster.KMeans(init='k-means++', n_clusters=n_digits, random_state=random_state)

        # fit the x_train data to the model
        clf.fit(x_train)

        if show:
            #  create the figure with a size of 8x3 inches
            fig = plt.figure(figsize=(8, 4))

            # Add title
            fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

            # For all labels (0-9)
            for i in range(10):
                # Initialize subplots in a grid of 2X5, at i+1th position
                ax = fig.add_subplot(2, 5, 1 + i)
                # Display images
                ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest")
                # Don't show the axes
                plt.axis('off')

            # Show the plot
            plt.show()

        # predict the labels for x_test
        y_pred = clf.predict(x_test)

        # print out the first 50 predicted and test values
        print("Predicted Values:\n",y_pred[:50])
        print("Target Values:\n",y_test[:50])
        print("Shape of Data:\n",clf.cluster_centers_.shape)

        # Create an isomap and fit the `digits` data to it
        x_iso = Isomap(n_neighbors=10).fit_transform(x_train)

        # Compute cluster centers and predict cluster index for each sample
        clusters = clf.fit_predict(x_train)

        if show:
            # Create a plot with subplots in a grid of 1X2
            fig = plt.figure(1, (8, 4))
            gs = gridspec.GridSpec(1, 2)
            ax = [fig.add_subplot(ss) for ss in gs]

            # Adjust layout
            fig.suptitle('Predicted Versus Training Labels(ISOMAP)', fontsize=14, fontweight='bold')

            # Add scatterplots to the subplots
            ax[0].scatter(x_iso[:, 0], x_iso[:, 1], c=clusters, edgecolors='black')
            ax[0].set_title('Predicted Training Labels')
            ax[1].scatter(x_iso[:, 0], x_iso[:, 1], c=y_train, edgecolors='black')
            ax[1].set_title('Actual Training Labels')

            gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

            # Show the plots
            plt.show()

        # Model and fit the `digits` data to the PCA model
        x_pca = PCA(n_components=2).fit_transform(x_train)

        # Compute cluster centers and predict cluster index for each sample
        clusters = clf.fit_predict(x_train)

        if show:
            # Create a plot with subplots in a grid of 1X2
            fig = plt.figure(1, (8, 4))
            gs = gridspec.GridSpec(1, 2)
            ax = [fig.add_subplot(ss) for ss in gs]

            # Adjust layout
            fig.suptitle('Predicted Versus Training Labels (PCA)', fontsize=14, fontweight='bold')
            fig.subplots_adjust(top=0.85)

            # Add scatterplots to the subplots
            ax[0].scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, edgecolors='black')
            ax[0].set_title('Predicted Training Labels')
            ax[1].scatter(x_pca[:, 0], x_pca[:, 1], c=y_train, edgecolors='black')
            ax[1].set_title('Actual Training Labels')

            gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

        # Show the plots
        plt.show()

        # Print out the confusion matrix to see how the model is incorrect
        print("Classification Report:\n",metrics.classification_report(y_test, y_pred))
        print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))

        # So looking at these numbers we can see that the kmeans model is not a good fit for our problem
        # this means that we must pick a different model for our data
        print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
        print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (clf.inertia_,
                 homogeneity_score(y_test, y_pred),
                 completeness_score(y_test, y_pred),
                 v_measure_score(y_test, y_pred),
                 adjusted_rand_score(y_test, y_pred),
                 adjusted_mutual_info_score(y_test, y_pred),
                 silhouette_score(x_test, y_pred, metric='euclidean')))


    def svc_model(self, test_size, random_state, show=None):
        # splitting the data into training and testing sets
        # typically 3/4 of the data is used to train, 1/4 of the data is used to test
        # x is the data you are testing : y is the target values of the corresponding data
        x_train, x_test, y_train, y_test, images_train, images_test = train_test_split(self.data, self.target,
                                                                                       self.images,
                                                                                       test_size=test_size,
                                                                                       random_state=random_state)
        # gets the number of training features
        n_samples, n_features = x_train.shape

        # print out the number of samples and features
        print("# of training samples: ",n_samples)
        print("# of training features: ",n_features)

        # set the parameter candidates
        parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]

        # create a classifier with the parameter candidates
        clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

        # train the classifier on training data
        clf.fit(x_train, y_train)

        # Print out the results
        print('Best score for training data:', clf.best_score_)
        print('Best `C`:', clf.best_estimator_.C)
        print('Best kernel:', clf.best_estimator_.kernel)
        print('Best `gamma`:', clf.best_estimator_.gamma)

        # train the svc model with the best
        svc_model = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma)
        svc_model.fit(x_train, y_train)
        print('Score :', svc_model.score(x_test, y_test))

        # print out the predicted values against the test values
        print("Predicted Values:\n",svc_model.predict(x_test))
        print("Target Values:\n",y_test)

        # Assign the predicted values to `predicted`
        predicted = svc_model.predict(x_test)

        if show:
            # Zip together the `images_test` and `predicted` values in `images_and_predictions`
            images_and_predictions = list(zip(images_test, predicted))

            # For the first 4 elements in `images_and_predictions`
            for index, (image, prediction) in enumerate(images_and_predictions[:4]):
                # Initialize subplots in a grid of 1 by 4 at positions i+1
                plt.subplot(1, 4, index + 1)
                # Don't show axes
                plt.axis('off')
                # Display images in all subplots in the grid
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                # Add a title to the plot
                plt.title('Predicted: ' + str(prediction))

            # Show the plot
            plt.show()

        print("Classification Report:\n",metrics.classification_report(y_test, predicted))
        print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, predicted))

        # Create an isomap and fit the `digits` data to it
        x_iso = Isomap(n_neighbors=10).fit_transform(x_train)

        # Compute cluster centers and predict cluster index for each sample
        predicted = svc_model.predict(x_train)

        if show:
            # Create a plot with subplots in a grid of 1X2
            fig = plt.figure(1, (8, 4))
            gs = gridspec.GridSpec(1, 2)
            ax = [fig.add_subplot(ss) for ss in gs]

            # Adjust the layout
            fig.subplots_adjust(top=0.85)

            # Add title
            fig.suptitle('Predicted Versus Actual Labels', fontsize=14, fontweight='bold')

            # Add scatterplots to the subplots
            ax[0].scatter(x_iso[:, 0], x_iso[:, 1], c=predicted, edgecolors='black')
            ax[0].set_title('Predicted labels')
            ax[1].scatter(x_iso[:, 0], x_iso[:, 1], c=y_train, edgecolors='black')
            ax[1].set_title('Actual Labels')

            gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

            # Show the plot
            plt.show()
