import csv
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

lines: list
train_labels: np.array
train_features: np.ndarray
validation_labels: np.array
validation_features: np.ndarray
test_features: np.ndarray
test_labels: np.array  # predictions


def read_dataset(file: str, is_train=True):
    global lines, train_labels, train_features, test_features, test_labels, validation_labels, validation_features
    if is_train:
        with open(file, "r") as csv_file:
            train_data = csv.reader(csv_file)
            lines = [dt for dt in train_data]
        classes = []
        classes2 = []
        train_list, validation_list = split_training_set(lines)
        for i in range(1, len(train_list)):
            if len(train_list[i][-1]) == 5:
                classes.append(1)
            else:
                classes.append(0)
        for j in range(1, len(validation_list)):
            if len(validation_list[j][-1]) == 5:
                classes2.append(1)
            else:
                classes2.append(0)
        train_labels = np.array(classes)
        train_features = np.array(
            [" ".join(train_list[i][:-1]) for i in range(1, len(train_list))])  # remove the last column
        validation_labels = np.array(classes2)  # used for back-testing
        validation_features = np.array([" ".join(validation_list[i][:-1]) for i in range(1, len(validation_list))])
    else:
        with open(file, "r") as csv_file:
            test_data = csv.reader(csv_file)
            lines = [dt for dt in test_data]
        test_features = np.array(([" ".join(lines[i]) for i in range(1, len(lines))]))
        test_labels = np.zeros(len(lines[1:]))


def split_training_set(x: list):
    # use a 75 to 25 ratio for training and validation sets.
    np.random.shuffle(x)
    size = int(len(x) * .75)
    return x[:size], x[size:]


# read_dataset("adult_train_CS584.csv")
# read_dataset("adult_test_CS584.csv", False)
# print(test_labels)
class Classifiers:

    def __init__(self, train_file, test_file):
        self.start = time.process_time()
        read_dataset(train_file)  # read train data
        read_dataset(test_file, False)  # read test data
        self.X_train = None  # training matrix
        self.X_test = None  # validation for back testing
        self.Y_test = None  # testing matrix
        self.predictions = None
        print("train_features {}".format(train_features.shape))
        print("validation_features {}".format(validation_features.shape))
        print("test_features {}".format(test_features.shape))
        self.f1_scores = []

    def vectorize(self):
        print("vectorizing the dataset...")
        # use_idf = false + 10000 max features ----> 0.84 and true + 16000 ----> 0.85
        tfidf = TfidfVectorizer(norm='l2', use_idf=True, max_features=16000)
        self.X_train = tfidf.fit_transform(train_features)
        self.X_test = tfidf.transform(validation_features)
        self.Y_test = tfidf.transform(test_features)
        print("Training vector shape = {}".format(self.X_train.shape))
        print("Validation vector shape = {}".format(self.X_test.shape))
        print("Testing vector shape = {}".format(self.Y_test.shape))
        print("-" * 75)
        print("Calculating SVD....")
        # changed from 100 to 426 (max components)
        # runtime went up to 474.81s ----> 0.87
        svd = TruncatedSVD(n_components=426)
        self.X_train = svd.fit_transform(self.X_train)
        self.X_test = svd.transform(self.X_test)
        self.Y_test = svd.transform(self.Y_test)
        print("Training matrix shape = {}".format(self.X_train.shape))
        print("Training matrix shape = {}".format(self.X_test.shape))
        print("Training matrix shape = {}".format(self.Y_test.shape))
        print("Time elapsed to clean-up data = {:.2f}".format(time.process_time() - self.start))
        print("-" * 75)

    def train(self, alg='svm', output="prediction.txt"):
        if self.X_train is None:
            self.vectorize()
        self.start = time.process_time()  # restart to measure runtime for classifiers
        if alg == 'svm':
            print("implementing SVM...")
            # changed SVC C from 10 to 1
            svm = SVC(kernel='rbf', C=100, gamma=1e1, tol=1e-5, probability=True, random_state=0, break_ties=True)
            svm.fit(self.X_train, train_labels)
            val_prediction = svm.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[0]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[1]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[2]))
            print("predictor initialized...")
            self.predictions = svm.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for SVM = {:.2f}".format(time.process_time() - self.start))
        elif alg == 'dt':  # decision tree
            print("implementing decision tree...")
            predictor = SVC()
            decision_tree = GridSearchCV(predictor, {'kernel': ('linear', 'rbf'), 'C': [.001, 1000], 'gamma': [10]})
            decision_tree.fit(self.X_train, train_labels)
            val_prediction = decision_tree.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[0]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[1]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[2]))
            print("predictor initialized...")
            self.predictions = decision_tree.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for SVM = {:.2f}".format(time.process_time() - self.start))
        elif alg == 'knn':  # KNN
            print("implement K nearest neighbors ...")
            k = 7
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, train_labels)
            val_prediction = knn.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[0]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[1]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[2]))
            print("predictor initialized...")
            self.predictions = knn.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for SVM = {:.2f}".format(time.process_time() - self.start))
            pass
        with open(output, "w") as file:
            for score in self.predictions:
                file.write(str(score) + "\n")

    @staticmethod
    def get_f1_score(truth, predict, mode='micro'):
        return f1_score(truth, predict, average=mode)


if __name__ == '__main__':
    classifier = Classifiers("adult_train_CS584.csv", "adult_test_CS584.csv")
    classifier.vectorize()
    classifier.train()
