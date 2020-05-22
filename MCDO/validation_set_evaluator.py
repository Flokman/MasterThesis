import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
import sklearn.metrics
import datetime
import numpy as np

class ValidationSetEvaluator(Callback):
    def __init__(self, X_val, Y_val, classes):
        self.classes = classes
        self.X_val = X_val
        self.Y_val = Y_val # one hot labels
        self.labels = np.argmax(self.Y_val, axis=1) # regular label indexes

    def on_train_begin(self, logs={}):
        self.history = []
        self.min_max = {}
        self.started = datetime.datetime.now().strftime("%Y-%m-%d_(%H.%M.%S)")
        self.log = ''
        self.epoch = 0
        plt.ion()
        plt.show()

    # def calculate_kappa(self, confusion_matrix):
    #     total_correct = sum(np.diag(confusion_matrix))
    #     total = sum(sum(confusion_matrix))
    #     accuracy = total_correct / total
    #     expected = 1/(total*total) * sum([sum(confusion_matrix[:, c]) * sum(confusion_matrix[c, :]) for c in range(len(self.classes))])
    #     kappa = (accuracy - expected) / (1 - expected)
    #     return kappa

    def get_class_metrics(self, confusion_matrix):
        class_statistics = []
        for c in range(len(self.classes)):
            class_total_count = sum(confusion_matrix[c, :])
            labeled_count = sum(confusion_matrix[:, c])
            recall = confusion_matrix[c, c] / class_total_count # divide correctly classified by all of class
            precision = confusion_matrix[c, c] / labeled_count # divide correctly classified by all labeled as this class
            class_statistics.append({'recall': recall, 'precision': precision})
        return class_statistics

    def calculate_metrics(self, evaluation_results, prediction):
        confusion_matrix = sklearn.metrics.confusion_matrix(self.labels, prediction, labels=range(len(self.classes))).astype(np.float64)
        #kappa = self.calculate_kappa(confusion_matrix)
        accuracy_control = sum(np.diag(confusion_matrix)) / sum(sum(confusion_matrix))
        metrics = {'loss': evaluation_results['loss'], 'accuracy': accuracy_control, 'kappa': evaluation_results['cohen_kappa']}
        class_metrics = self.get_class_metrics(confusion_matrix)
        return (metrics, class_metrics, confusion_matrix)

    def track_min_max(self, metrics):
        for label, value in metrics.items():
            if label not in self.min_max:
                self.min_max[label] = value
            else:
                # keep minimum for loss
                if 'loss' in label:
                    self.min_max[label] = min(self.min_max[label], value)
                else:
                    self.min_max[label] = max(self.min_max[label], value)

    def do_validation(self):
        evaluation_results = self.model.evaluate(x = self.X_val, y = self.Y_val)
        evaluation_results = dict(zip(self.model.metrics_names, evaluation_results))
        prediction = np.argmax(self.model.predict(self.X_val), axis=1)
        return (evaluation_results, prediction)

    def print_results(self, metrics, class_metrics, confusion_matrix, min_max):
        result = '\n'
        result += '\nepoch ' + str(self.epoch)
        result += '\n' + np.array2string(confusion_matrix)
        for c in range(len(self.classes)):
            result += '\nclass ' + str(self.classes[c])
            for label, value in class_metrics[c].items():
                result += ' ' + label + ': ' + str(value)
        result += '\n'
        for label, value in metrics.items():
            if 'kappa2' is label or  'kappa' is label or 'accuracy' is label:
                result += label + ': ' + str(value) + ' '
        result += '\nbest:\n'
        for label, value in min_max.items():
            if 'kappa' is label or 'accuracy' is label:
                result += label + ': ' + str(value) + ' '
        self.log += result
        print(result)

    def plot(self, title, y_label, n_plots, subplot, x_values, y_values_validation, y_values_train):
        plt.subplot(n_plots, 1, subplot)
        plt.title(title)
        # plt.axis([0,100,0,1])
        #for (colour, y_values) in list_y_value_tuples:
        plt.plot(x_values, y_values_train, color='b')
        plt.plot(x_values, y_values_validation, color='r')
        plt.ylabel(y_label)

        red_patch = mpatches.Patch(color='red', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch], loc=4)

    def plot_metrics(self, to_plot):
        epochs = [x for x in range(self.epoch+1)]
        n_plots = len(to_plot)
        counter = 1
        for (name, train_name, validation_name) in to_plot:
            values_train = [metric[train_name] for metric in self.history]
            values_validation = [metric[validation_name] for metric in self.history]
            self.plot(name, name, n_plots, counter, epochs, values_validation, values_train)
            counter += 1

        plt.draw()
        plt.pause(0.001)

        return

        count_subplots = 0
        count_subplots += 1
        plt.subplot(self.num_subplots, 1, count_subplots)
        plt.title('Accuracy')
        # plt.axis([0,100,0,1])
        plt.plot(epochs, self.val_acc, color='r')
        plt.plot(epochs, self.acc, color='b')
        plt.ylabel('accuracy')

        red_patch = mpatches.Patch(color='red', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')

        plt.legend(handles=[red_patch, blue_patch], loc=4)

        # loss plot
        count_subplots += 1
        plt.subplot(self.num_subplots, 1, count_subplots)
        plt.title('Loss')
        # plt.axis([0,100,0,5])
        plt.plot(epochs, self.val_loss, color='r')
        plt.plot(epochs, self.loss, color='b')
        plt.ylabel('loss')

        red_patch = mpatches.Patch(color='red', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')

        plt.legend(handles=[red_patch, blue_patch], loc=4)

        plt.draw()
        plt.pause(0.001)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch

        (evaluation_results, prediction) = self.do_validation()
        (metrics, class_metrics, confusion_matrix) = self.calculate_metrics(evaluation_results, prediction)
        metrics['train_kappa'] = logs.get('cohen_kappa')
        metrics['train_loss'] = logs.get('loss')
        metrics['train_accuracy'] = logs.get('categorical_accuracy')
        self.track_min_max(metrics)
        self.print_results(metrics, class_metrics, confusion_matrix, self.min_max)
        # add metrics to list at each epoch
        self.history.append(metrics)

        to_plot = [('kappa', 'train_kappa', 'kappa'),
                   ('loss', 'train_loss', 'loss'),
                   ('accuracy', 'train_accuracy', 'accuracy')]
        self.plot_metrics(to_plot)

        return

    def on_train_end(self, logs={}):
        basefilename = 'result_' + self.started
        plt.savefig(basefilename + ".png")
        with open(basefilename + ".txt", "w") as text_file:
            text_file.write(self.log)
        # this will prevent program from exiting while plot is open
        plt.ioff()
        plt.show()
