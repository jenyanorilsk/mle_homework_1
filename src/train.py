import configparser
import os
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from logger import Logger

SHOW_LOG = True

class SpamClassifier():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), '..', 'config.ini')
        self.config.read(self.config_path)
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.X_train = self.X_train['text']
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.y_train = self.y_train['classname']
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.X_test = self.X_test['text']
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.y_test = self.y_test['classname']
        self.project_path = os.path.join(os.getcwd(), "..", "experiments")
        self.model_path = os.path.join(self.project_path, "model.sav")
        self.log.info("SpamClassifier is ready")

    def train_model(self, predict=False) -> bool:
        classifier = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print('accuracy:', accuracy_score(self.y_test, y_pred))
            print('confusion matrix:')
            print(confusion_matrix(self.y_test, y_pred))
            print('classification report:')
            print(classification_report(self.y_test, y_pred))
        params = {'path': self.model_path}
        return self.save_model(classifier, self.model_path, "MODEL", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove(self.config_path)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    model = SpamClassifier()
    model.train_model(predict=True)
