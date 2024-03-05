import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class AutoencoderUtils():

    def __init__(self, log, percentile=0.85):
        self.log = log
        self.percentile = percentile

        self.log.debug("DEBUG: AutoencoderUtils init")

    def getData(self, train_file, anomalous_file):

        x_train_df = pd.read_csv(train_file)
        x_abnormal_df = pd.read_csv(anomalous_file, header=None)

        x_train = x_train_df.values
        x_anomaly = x_abnormal_df.values

        self.log.debug("DEBUG:x_train:" + str(len(x_train)))
        self.log.debug("DEBUG:x_anomaly:" + str(len(x_anomaly)))

        x_train, x_test = train_test_split(x_train, test_size=0.2)

        return x_train, x_test, x_anomaly

    def getThreasholdTrain(self, autoencoder, x):
        predictions = autoencoder.predict(x)
        mse = np.mean(np.power(x - predictions, 2), axis=1)

        threshold = np.quantile(mse, self.percentile)

        return threshold

    def plotLoss(self, autoencoder, x_test, y_test, threshold, model_name):

        predictions = autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)

        error_df = pd.DataFrame(list(zip(list(mse.values.reshape(1, len(mse))[0]),
                                         list(y_test.values.reshape(1, len(y_test))[0]))),
                                columns=['reconstruction_error', 'true_class'])

        self.log.info('************************** Evalaution Result **************************')
        self.log.info(f'Threshold= {threshold}')

        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

        conf_matrix = confusion_matrix(error_df.true_class, y_pred)

        tn, fp, fn, tp = conf_matrix.ravel()

        precision = 1. * tp / (tp + fp)
        recall = 1. * tp / (tp + fn)
        f1 = (2 * recall * precision) / (recall + precision)
        accuracy = 1. * (tp + tn) / (tp + tn + fp + fn)

        self.log.info('TP:' + str(np.round(tp, 3)))
        self.log.info('FP:' + str(np.round(fp, 3)))
        self.log.info('TN:' + str(np.round(tn, 3)))
        self.log.info('FN:' + str(np.round(fn, 3)))
        self.log.info('Accuracy:' + str(np.round(accuracy, 3)))
        self.log.info('Precision:' + str(np.round(precision, 3)))
        self.log.info('Recall:' + str(np.round(recall, 3)))
        self.log.info('F1:' + str(np.round(f1, 3)))

        groups = error_df.groupby('true_class')

        # plot the loss
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=2, linestyle='',
                    label="Abnormal data" if name == 1 else "Normal data", color='red' if name == 1 else 'orange')
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="green",
                  zorder=100, label='Threshold=' + str(np.round(threshold, 3)))
        ax.legend()
        plt.title(model_name + " | F1 = " + str(np.round(f1, 3)))
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data")

        plt.savefig(Path('figs') / 'f1.png')
        self.log.info(f"F1 plot saved in {Path('figs') / 'f1.png'}")

    def driver(self, model, model_name, train, anomaly):

        x_train, x_test, x_abnormal = self.getData(train, anomaly)

        x_test = x_test[:len(x_abnormal)]

        threshold = self.getThreasholdTrain(model, pd.DataFrame(x_train))

        x_combine = pd.DataFrame(np.concatenate([x_test, x_abnormal], axis=0))
        y_test = pd.DataFrame([0 for _ in range(len(x_test))]+[1 for _ in range(len(x_abnormal))])

        self.plotLoss(model, x_combine, y_test, threshold, model_name)

        self.log.info(f'Threshold = {np.round(threshold,3)}')