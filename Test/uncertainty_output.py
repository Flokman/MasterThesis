import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score
from statistics import mean

class Uncertainty_output:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    

    def categorical_cross(self, y_true, y_pred, from_logits=False):
        # Determine the loss by calculating the categorical crossentropy on only the first num_classes outputs 
        y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        y_true_cat = y_true[:, :self.num_classes]
        y_pred_cat = y_pred[:, :self.num_classes]
        cat_loss = K.categorical_crossentropy(y_true_cat, y_pred_cat, from_logits=from_logits)
        
        return cat_loss


    def categorical_error(self, y_true, y_pred, from_logits=False):
        # Determine the loss by calculating the categorical crossentropy on only the first num_classes outputs
        # and the mean squared error on the error outputs
        y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        y_true_cat = y_true[:, :self.num_classes]
        y_pred_cat = y_pred[:, :self.num_classes]
        cat_loss = K.categorical_crossentropy(y_true_cat, y_pred_cat, from_logits=from_logits)

        y_pred_cat_abs = K.abs(y_pred_cat)
        y_true_error = K.square(y_pred_cat_abs - y_true_cat)
        y_pred_error = y_pred[:, self.num_classes:]
        error_loss = K.mean(K.square(y_pred_error - y_true_error), axis=-1)
        total_loss = cat_loss + error_loss

        return total_loss

    
    def create_uncertainty_model(self, model):
        # Replace last layer of network by uncertainty layer
        # Note: doesn't seem to work for some networks
        all_layers = [l for l in model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers) - 1):
            x = all_layers[i](x)

        classification = layers.Dense(self.num_classes, activation='softmax')(x)
        error_output = layers.Dense(self.num_classes, activation='linear')(x)

        out = layers.concatenate([classification, error_output])

        # Creating new model
        uncertainty_model = Model(inputs=all_layers[0].input, outputs=out)
        return uncertainty_model
    

    def create_uncertainty_layer(self, x):
        # Adds an uncertainty layer as last layer to the network

        classification = layers.Dense(self.num_classes, activation='softmax')(x)
        error_output = layers.Dense(self.num_classes, activation='linear')(x)

        out = layers.concatenate([classification, error_output])

        return out


    def convert_output_to_uncertainty(self, prediction):
        #Convert error prediction to uncertainty
        for ind, pred in enumerate(prediction):
            predictions = pred[:self.num_classes]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = np.abs(pred[self.num_classes:])

            for i in range(0, self.num_classes):
                pred_var = uncertainties[i]
                if i == highest_pred_ind:
                    # Highest predicted class, so error as if true
                    true_error = pow((predictions[i] - 1), 2)
                else:
                    # Not highest predicted class, so error as if false
                    true_error = pow((predictions[i]), 2)
                uncertainties[i] = abs(true_error - pred_var)
            prediction[ind][self.num_classes:] = uncertainties   
         
        return prediction


    def results_if_label(self, org_data_prediction, y_test, scatter = False, name = ''):
        # score on the test images (if label avaialable)
        true_labels = [np.argmax(i) for i in y_test]
        wrong = 0
        correct = 0
        # Only when correctly predict, info of true class
        correct_unc = []
        correct_prob = []
        # Only when wrongly predicted, info of highest wrong pred
        high_wrong_unc = []
        high_wrong_prob = []
        # Only when wrongly predicted, info of true class
        true_wrong_unc = []
        true_wrong_prob = []        
        # Info of all incorrect classes
        all_wrong_unc = []
        all_wrong_prob = []
        # Info of all not true label classes
        not_true_label_unc = []
        not_true_label_prob = []
        # Info of all classes
        all_probabilities = []
        all_uncertainties = []

        for ind, pred in enumerate(org_data_prediction):
            true_label = true_labels[ind]
            predictions = pred[:self.num_classes]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = pred[self.num_classes:]

            for class_ind, (prob, unc) in enumerate(zip(predictions, uncertainties)):
                all_probabilities.append(prob)
                all_uncertainties.append(unc)

                if class_ind == true_label:
                    if highest_pred_ind == true_label:
                        correct += 1
                        correct_unc.append(unc)
                        correct_prob.append(prob)

                    else:
                        wrong += 1
                        true_wrong_unc.append(unc)
                        true_wrong_prob.append(prob)

                        high_wrong_unc.append(uncertainties[highest_pred_ind])
                        high_wrong_prob.append(predictions[highest_pred_ind])

                        all_wrong_unc.append(unc)
                        all_wrong_prob.append(prob)                    

                else:
                    all_wrong_unc.append(unc)
                    all_wrong_prob.append(prob)

                    not_true_label_unc.append(unc)
                    not_true_label_prob.append(prob)                    
                    

        acc = accuracy_score(y_test.argmax(axis=1), org_data_prediction[:, :self.num_classes].argmax(axis=1))
        print("Accuracy on original test dataset: {:.1%}".format(acc))

        confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=org_data_prediction[:, :self.num_classes].argmax(axis=1),
                                        num_classes=self.num_classes)
        print(confusion)

        print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(correct+wrong))*100))
        print("")
        print("Mean probability on true label of original test dataset when correctly predicted = {:.2%}".format(mean(correct_prob)))
        print("Mean uncertainty on true label of original test dataset when correctly predicted = {:.2%}".format(mean(correct_unc)))
        print("Mean probability on true label of original test dataset when wrongly predicted = {:.2%}".format(mean(true_wrong_prob))) 
        print("Mean uncertainty on true label of original test dataset when wrongly predicted = {:.2%}".format(mean(true_wrong_unc)))    

        print("")
        print("Mean probability on highest predicted on original test dataset when wrong = {:.2%}".format(mean(high_wrong_prob))) 
        print("Mean uncertainty on highest predicted on original test dataset when wrong = {:.2%}".format(mean(high_wrong_unc)))

        print("")
        print("Mean probability on all not true label on original test dataset = {:.2%}".format(mean(not_true_label_prob))) 
        print("Mean uncertainty on all not true label on original test dataset = {:.2%}".format(mean(not_true_label_unc)))
        
        if scatter:
            self.scatterplot(correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, 'ErrorOutput', name)


    def scatterplot(self, correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, methodname, own_or_new):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        
        if correct_prob == None:
            print("creating scatterplot")
            plt.clf()
            plt.style.use("ggplot")
            plt.xlim(0,1.05)
            plt.scatter(high_wrong_prob, high_wrong_unc, c='r')
            plt.legend()
            plt.xlabel('probability')
            plt.ylabel('uncertainty')
            plt.title('Scatterplot for {} on {}'.format(methodname, own_or_new))
            plt.savefig('{}_scatter_{}.png'.format(methodname, own_or_new))
            plt.clf()

        else:
            print("creating scatterplot")
            plt.clf()
            plt.style.use("ggplot")
            plt.xlim(0,1.05)
            plt.scatter(high_wrong_prob, high_wrong_unc, c='r', label='Wrongly predicted')
            plt.scatter(correct_prob, correct_unc, c='g', label='Correctly predicted')
            plt.legend()
            plt.xlabel('probability')
            plt.ylabel('uncertainty')
            plt.title('Scatterplot for {} on {}'.format(methodname, own_or_new))
            plt.savefig('{}_scatter_{}.png'.format(methodname, own_or_new))
            plt.clf()


    def MSE_scatterplot(self, MSE, UNC, methodname, own_or_new):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        
        print("creating MSE scatterplot")
        plt.clf()
        plt.style.use("ggplot")
        plt.scatter(MSE, UNC, c='b', label='Correctly predicted')
        plt.legend()
        plt.xlabel('MSE')
        plt.ylabel('Predicted Error')
        plt.title('Scatterplot for {} on {}'.format(methodname, own_or_new))
        plt.savefig('{}_MSE_scatter_{}.png'.format(methodname, own_or_new))
        plt.clf()


    def results_if_no_label(self, new_images_predictions, details_per_example = False, name = ''):
        highest_probs = []
        highest_probs_uncs = []
        not_highest_probs = []
        not_highest_probs_uncs = []
        all_probs = []
        all_uncs = []

        for ind, pred in enumerate(new_images_predictions):
            predictions = pred[:self.num_classes]
            uncertainties = pred[self.num_classes:]
            highest_pred_ind = np.argmax(predictions)

            for class_ind, (prob, unc) in enumerate(zip(predictions, uncertainties)):
                all_probs.append(prob)
                all_uncs.append(unc)

                if class_ind == highest_pred_ind:
                    highest_probs.append(prob)
                    highest_probs_uncs.append(unc)
                
                else:
                    not_highest_probs.append(prob)
                    not_highest_probs_uncs.append(unc)

        print("")
        print("Mean probability on highest predicted class of new data = {:.2%}".format(mean(highest_probs)))
        print("Mean uncertainty on highest predicted class of new data = {:.2%}".format(mean(highest_probs_uncs)))
        print("Mean probability on not highest predicted class of new data = {:.2%}".format(mean(not_highest_probs)))
        print("Mean uncertainty on not predicted classes of new data = {:.2%}".format(mean(not_highest_probs_uncs)))


        if details_per_example:
            for ind, pred in enumerate(new_images_predictions):
                predictions = pred[:self.num_classes]
                uncertainties = pred[self.num_classes:]
                highest_pred_ind = np.argmax(predictions)

                print("Predicted class: {}, Uncertainty of prediction: {:.2%}".format(highest_pred_ind, uncertainties[highest_pred_ind]))

                for class_ind, (prob, unc) in enumerate(zip(predictions, uncertainties)):
                    print("Class: {}; Probability: {:.1%}; Uncertainty: {:.2%} ".format(class_ind, prob, unc))
        
        if scatter:
            self.scatterplot(None, None, all_probs, all_uncs, 'ErrorOutput', name)


    def MSE(self, org_data_prediction, y_test, scatter = False, name = ''):
        true_labels = [np.argmax(i) for i in y_test]
        SSE = []
        Hi_Error = []

        for ind, pred in enumerate(org_data_prediction):
            true_label = true_labels[ind]
            predictions = pred[:self.num_classes]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = pred[self.num_classes:]

            Squared_Error = pow((predictions[highest_pred_ind] - 1), 2)
            SSE.append(Squared_Error)
            Hi_Error.append(uncertainties[highest_pred_ind])
        
        MSE = mean(SSE)
        self.MSE_scatterplot(MSE, Hi_Error, 'ErrorOutput', name)