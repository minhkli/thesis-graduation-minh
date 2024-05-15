from sklearn import svm
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def calculate_accuracy(y_true, y_pred):
    total_samples = len(y_true)
    correct_predictions = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

with open("train_vege.pkl", "rb") as f:
    X, y = pkl.load(f)

def svm_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = calculate_accuracy(y_test, y_pred)
    # print("Accuracy:", accuracy)
    print("SVM Score:", clf.score(X_test,y_test))
    pkl.dump(clf, open("svm_model.pickle", "wb"))

    # Lấy ra loss cho tập huấn luyện
    train_loss = -np.mean(clf.score(X_train, y_train))

    # Lấy ra accuracy cho tập huấn luyện và tập kiểm tra
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("train loss: ", train_loss)
    print("train accuracy: ", train_accuracy)
    print("test accuracy: ", test_accuracy)


def Kneighbor(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)

    # Lấy ra loss cho tập huấn luyện
    train_loss = -np.mean(nca_pipe.score(X_train, y_train))

    # Lấy ra accuracy cho tập huấn luyện và tập kiểm tra
    train_accuracy = nca_pipe.score(X_train, y_train)
    test_accuracy = nca_pipe.score(X_test, y_test)
    print("train loss: ", train_loss)
    print("train accuracy: ", train_accuracy)
    print("test accuracy: ", test_accuracy)



svm_model(X,y)
Kneighbor(X,y)