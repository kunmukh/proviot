import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class AutoencoderUtils():

    def __init__(self):
        print("DEBUG: AutoencoderUtils init")

    def getData(self, fileNameTrain, fileNameAbnormal):

        x_train_df = pd.read_csv(fileNameTrain)
        x_abnormal_df = pd.read_csv(fileNameAbnormal, header=None)

        x_train = x_train_df.values
        x_abnormal = x_abnormal_df.values

        print("DEBUG:x_train:" + str(len(x_train)))
        print("DEBUG:x_abnormal:" + str(len(x_abnormal)))

        x_train, x_test = train_test_split(x_train, test_size=0.2)

        return x_train, x_test, x_abnormal

    def showloss(self, x_test, x_abnormal, model):
        x_concat = np.concatenate([x_test, x_abnormal], axis=0)
        losses = []
        for x in x_concat:
            x = np.expand_dims(x, axis=0)
            loss = model.test_on_batch(x, x)
            losses.append(loss[0])

        plt.plot(len(losses), losses, linestyle='-', linewidth=1, label="normal data", color='blue')
        plt.plot(len(losses), losses, linestyle='-', linewidth=1, label="anomaly data", color='red')

        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")

    def plotLoss(self, autoencoder, X_test, y_test, threshold, modelName):

        predictions = autoencoder.predict(X_test)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)

        error_df = pd.DataFrame(list(zip(list(mse.values.reshape(1, len(mse))[0]),
                                         list(y_test.values.reshape(1, len(y_test))[0]))),
                                columns=['reconstruction_error', 'true_class'])

        print('\nLoss Test **************************')
        print('threshold', threshold)

        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

        conf_matrix = confusion_matrix(error_df.true_class, y_pred)

        tn, fp, fn, tp = conf_matrix.ravel()

        precision = 1. * tp / (tp + fp)
        recall = 1. * tp / (tp + fn)
        f1 = (2 * recall * precision) / (recall + precision)
        accuracy = 1. * (tp + tn) / (tp + tn + fp + fn)

        print('TP:' + str(tp))
        print('FP:' + str(fp))
        print('TN:' + str(tn))
        print('FN:' + str(fn))
        print('Accuracy:' + str(accuracy))
        print('Precision:' + str(precision))
        print('Recall:' + str(recall))
        print('F1:' + str(f1))

        groups = error_df.groupby('true_class')

        # plot the loss
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=2, linestyle='',
                    label="Abnormal data" if name == 1 else "Normal data", color='red' if name == 1 else 'orange')
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="green",
                  zorder=100, label='Threshold=' + str(np.round(threshold, 3)))
        ax.legend()
        plt.title(modelName + " Reconstruction error| Accuracy= " + str(accuracy))
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data")

    def getThreasholdTrain(self, autoencoder, X):

        predictions = autoencoder.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)

        PERCENTILE = 0.85
        threshold = np.quantile(mse, PERCENTILE)

        return threshold

    def driver(self, model, modelName, train, ano):

        x_train, x_test, x_abnormal = self.getData(train, ano)

        x_test = x_test[:len(x_abnormal)]
        x_abnormal = x_abnormal[:len(x_test)]

        threshold = self.getThreasholdTrain(model, pd.DataFrame(x_train))

        X_test = pd.DataFrame(np.concatenate([x_test, x_abnormal], axis=0))
        Y_test = pd.DataFrame([0 for _ in range(len(x_test))]+[1 for _ in range(len(x_abnormal))])

        self.plotLoss(model, X_test, Y_test, threshold, modelName)

        print('Threshold', threshold)
        plt.show()