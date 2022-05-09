import csv
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

lines: list
train_labels: np.array
train_features: np.ndarray
train_features_copy: np.ndarray
validation_labels: np.array
validation_features: np.ndarray
validation_features_copy: np.ndarray
test_features: np.ndarray
test_labels: np.array  # predictions


def read_dataset(file: str, is_train=True):
    global lines, train_labels, train_features, test_features, test_labels, validation_labels, validation_features, train_features_copy, validation_features_copy
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
        train_features_copy = np.array(train_list)
        train_features = np.array(
            [" ".join(train_list[i][:-1]) for i in range(1, len(train_list))])  # remove the last column
        validation_labels = np.array(classes2)  # used for back-testing
        validation_features_copy = np.array(validation_list)
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
        svd = TruncatedSVD(n_components=400)
        self.X_train = svd.fit_transform(self.X_train)
        self.X_test = svd.transform(self.X_test)
        self.Y_test = svd.transform(self.Y_test)
        print("Training matrix shape = {}".format(self.X_train.shape))
        print("validation matrix shape = {}".format(self.X_test.shape))
        print("Testing matrix shape = {}".format(self.Y_test.shape))
        print("Time elapsed to pre-process data = {:.2f}".format(time.process_time() - self.start))
        print("-" * 75)

    def train(self, alg='svm', output="prediction.txt"):
        if self.X_train is None:
            self.vectorize()
        self.start = time.process_time()  # restart to measure runtime for classifiers
        if alg == 'svm':
            print("training with SVM...")
            self.start = time.process_time()
            # changed SVC C from 10 to 1
            # changed C to 10 and gamma to 0.1 with random state = 42
            svm = SVC(kernel='rbf', C=10, gamma=0.1, tol=1e-3, probability=True, random_state=42, break_ties=True)
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
            print("Runtime for SVM = {:.2f} min".format((time.process_time() - self.start) / 60.))
        elif alg == 'dt':  # decision tree
            print("training with decision tree...")
            self.start = time.process_time()
            predictor = SVC()
            decision_tree = GridSearchCV(predictor,
                                         {'kernel': ('linear', 'rbf'), 'C': [.1, 10], 'gamma': [0.1], 'tol': [1e-3]})
            decision_tree.fit(self.X_train, train_labels)
            val_prediction = decision_tree.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[3]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[4]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[5]))
            print("predictor initialized...")
            self.predictions = decision_tree.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for decision tree = {:.2f} min".format((time.process_time() - self.start) / 60.))
        elif alg == 'knn':  # KNN
            print("training with K nearest neighbors ...")
            self.start = time.process_time()
            k = 3  # k=3 gives the highest prediction
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, train_labels)
            val_prediction = knn.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[6]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[7]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[8]))
            print("predictor initialized...")
            self.predictions = knn.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for knn = {:.2f} min".format((time.process_time() - self.start) / 60.))
        elif alg == 'bc':  # boosting classifier
            print("training with gradient boosting classifier...")
            self.start = time.process_time()
            # n_estimator of 100 ----> 0.87
            cg = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=2, random_state=0,
                                            max_features='auto')
            cg.fit(self.X_train, train_labels)
            val_prediction = cg.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[9]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[10]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[11]))
            print("predictor initialized...")
            self.predictions = cg.predict(self.Y_test)  # y_test
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for boosted classifier = {:.2f} min".format((time.process_time() - self.start) / 60.))
        elif alg == 'rf':
            print("training with random forest classifier...")
            self.start = time.process_time()
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(self.X_train, train_labels)
            val_prediction = rf.predict(self.X_test)  # back-testing
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'macro'))
            self.f1_scores.append(self.get_f1_score(validation_labels, val_prediction, 'weighted'))
            print("F1-score (micro) = {:.2f}".format(self.f1_scores[12]))
            print("F1-score (macro) = {:.2f}".format(self.f1_scores[13]))
            print("F1-score (weighted) = {:.2f}".format(self.f1_scores[14]))
            print("predictor initialized...")
            self.predictions = rf.predict(self.Y_test)
            print(self.predictions.shape)
            print(self.predictions)
            print("Runtime for random forest = {:.2f} min".format((time.process_time() - self.start) / 60.))
        else:
            raise ValueError  # unknown classifier passed

        with open(output, "w") as file:
            for score in self.predictions:
                file.write(str(score) + "\n")

    @staticmethod
    def get_f1_score(truth, predict, mode='micro'):
        return f1_score(truth, predict, average=mode)

    @staticmethod
    def plot(y):
        filename = "f1-score.png"
        x = np.array([1, 2, 3, 4, 5])
        x_ticks = ['svm', 'decision-tree', 'knn', 'boosted-classifier', 'random-forest']
        y1 = np.array([y[0], y[3], y[6], y[9], y[12]])  # micro
        y2 = np.array([y[1], y[4], y[7], y[10], y[13]])  # macro
        y3 = np.array([y[2], y[5], y[8], y[11], y[14]])  # weighted
        plt.xticks(x, x_ticks)
        plt.plot(x, y1, label='micro')
        plt.plot(x, y2, label='macro')
        plt.plot(x, y3, label='weighted')
        plt.legend(loc='best')
        plt.savefig(filename)

    @staticmethod
    def fairness_diagnosis():
        # col 7 and 8 (race, sex)
        biased_cols = np.array(["race", "sex"])
        race = []  # from training
        sex = []
        for person in train_features_copy:
            if person[7] == ' White' and person[-1] == " <=50K":
                race.append(1)
            elif person[7] != ' White' and person[-1] == " <=50K":
                race.append(0)
            if person[8] == " Male" and person[-1] == " <=50K":
                sex.append(1)
            elif person[8] != " Male" and person[-1] == " <=50K":
                sex.append(0)
        race = np.array(race)
        sex = np.array(sex)
        whites = np.array([i for i in race if i == 1])
        non_whites = np.array([i for i in race if i == 0])
        males = [i for i in sex if i == 1]
        females = [i for i in sex if i == 0]
        print("white: {:.2f}%\t non-white: {:.2f}%".format(np.sum(whites) / train_features_copy.shape[0] * 100.,
                                                             (int(race.shape[0]) - np.sum(whites)) /
                                                             train_features_copy.shape[0] * 100.))
        print("Male: {:.2f}%\t Female: {:.2f}%".format(np.sum(males) / train_features_copy.shape[0] * 100.,
                                                         (int(sex.shape[0]) - np.sum(males)) /
                                                         train_features_copy.shape[0] * 100.))


if __name__ == '__main__':
    classifier = Classifiers("adult_train_CS584.csv", "adult_test_CS584.csv")
    classifier.vectorize()
    classifier.fairness_diagnosis()
    # classifier.train(output='prediction-svm.txt')
    # classifier.train('dt', 'prediction_decision_tree.txt')
    # classifier.train('knn', 'prediction_knn.txt')
    # classifier.train('bc', "prediction_boosted_classifier.txt")
    # classifier.train('rf', "prediction_random_forest_classifier.txt")
    # classifier.plot(classifier.f1_scores)
