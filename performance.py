import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

class PerformanceViewer(object):
    
    def __init__(self):
        self.runs = 0
        self.plot_epoch = [0]
        # class 1 2 3 and overall
        self.plot_precision_class = [[0],[0],[0],[0]]
        self.plot_recall_class = [[0],[0],[0],[0]]
        self.plot_f1_class = [[0],[0],[0],[0]]
        
    def resetHistory(self):
        self.plot_epoch = [0]
        # class 1 2 3 and overall
        self.plot_precision_class = [[0],[0],[0],[0]]
        self.plot_recall_class = [[0],[0],[0],[0]]
        self.plot_f1_class = [[0],[0],[0],[0]]
    
    # this evaluation plots precision, recall and f1 after each epoch of the model
    # the metrics are evaluated on a test_dataset which is not used in training
    def evalModelTrainData(self):
        for i, precision_class in enumerate(self.plot_precision_class):
            plt.plot(self.plot_epoch, precision_class, label = self.convertIdToClass(i))
        plt.title("Precision per class")
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

        for i, recall_class in enumerate(self.plot_recall_class):
            plt.plot(self.plot_epoch, recall_class, label = self.convertIdToClass(i))
        plt.title("Recall per class")
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.show()

        for i, f1_class in enumerate(self.plot_f1_class):
            plt.plot(self.plot_epoch, f1_class, label = self.convertIdToClass(i))
        plt.title("F1 per class")
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()

        self.resetHistory()
   
    # same as evalModelTrainData, but plots for each class(b_s, i_s, o, overall) 
    # instead of metric (precision,recall,f1)
    def evalModelTrainDataClass(self):
        for i in range(len(self.plot_precision_class)):
            plt.plot(plot_epoch, self.plot_precision_class[i], label = "Precision")
            plt.plot(plot_epoch, self.plot_recall_class[i], label = "Recall")
            plt.plot(plot_epoch, self.plot_f1_class[i], label = "F1")
            plt.title("Class "+convertIdToClass(i))
            plt.xlabel('Epochs')
            plt.ylabel('Performance')
            plt.legend()
            plt.show()
        self.resetHistory()

    def convertIdToClass(self, id):
        if id == 0:
            return "B_S"
        elif id == 1:
            return "I_S"
        elif id == 2:
            return "O"
        else:
            return "overall class"
    
    # displays overall accuracy and loss of the model based on validation split data
    def basicEval(self, history):
        #this eval only works, if we use validation_split = 0.2 as additional parameter for model.fit()

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    # the performance evaulation by confusion matrix, enriched with detailed metrics scores
    # this metric works with new, unused data 
    def classicEval(self, model, ds, vw, x_test, y_test):
        # predict labels of test data

        ## evaluation
        # predict labels of test data
        y_test_pred_prob = model.predict(x_test)
        y_test_pred_sparse = y_test_pred_prob.argmax(axis=-1)
        y_test_pred = to_categorical(np.array(y_test_pred_sparse), num_classes=vw.n_tags)

        # compute confusion matrix
        conf_matrix = np.zeros((vw.n_tags, vw.n_tags))
        for i,tokens in enumerate(ds.test_tokens):
            for j,_ in enumerate(tokens):
                class_true = y_test[i,j].argmax()
                class_pred = y_test_pred[i,j].argmax()
                conf_matrix[class_true,class_pred] += 1
        names_rows = list(s+'_true' for s in vw.labelclass_to_id.keys())
        names_columns = list(s+'_pred' for s in vw.labelclass_to_id.keys())
        conf_matrix = pd.DataFrame(data=conf_matrix,index=names_rows,columns=names_columns)
        display(conf_matrix)

        # compute final evaluation measures
        precision_per_class = np.zeros((vw.n_tags,))
        recall_per_class = np.zeros((vw.n_tags,))
        for i in range(vw.n_tags):

            print("check gt 0")
            print(conf_matrix.values[i,i])
            print("")
            if conf_matrix.values[i,i] > 0:
                print("CLASS = "+str(i))
                precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
                print("precision")
                print(precision_per_class[i])
                recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])
                print("recall")
                print(recall_per_class[i])
                print("f1")
                f1_class = 2*(precision_per_class[i]*recall_per_class[i])/(precision_per_class[i]+recall_per_class[i])
                print(f1_class)
            else:
                print("no (0) precision or recall for class "+str(i))
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        f1 = 2*(precision*recall)/(precision+recall)
        print()
        print('Precision: '+str(precision))
        print('Recall: '+str(recall))
        print('F1-measure: '+str(f1))

        print("\n")
        
#callback for checking performance after each epoch
class TrainingEval(tf.keras.callbacks.Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    def __init__(self, model, x_test, y_test, vw, ds, performance):
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.vw = vw
        self.ds = ds
        self.performance = performance
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        #define data accuracy with modulo 2 = high, 10 = low
        if epoch % 2 == 1:
            self.performance.plot_epoch.append(epoch)
            ## evaluation
            # predict labels of test data
            y_test_pred_prob = self.model.predict(self.x_test)
            y_test_pred_sparse = y_test_pred_prob.argmax(axis=-1)
            y_test_pred = to_categorical(np.array(y_test_pred_sparse), num_classes=self.vw.n_tags)

            # compute confusion matrix
            conf_matrix = np.zeros((self.vw.n_tags, self.vw.n_tags))
            for i,tokens in enumerate(self.ds.test_tokens):
                for j,_ in enumerate(tokens):
                    class_true = self.y_test[i,j].argmax()
                    class_pred = y_test_pred[i,j].argmax()
                    conf_matrix[class_true,class_pred] += 1
            names_rows = list(s+'_true' for s in self.vw.labelclass_to_id.keys())
            names_columns = list(s+'_pred' for s in self.vw.labelclass_to_id.keys())
            conf_matrix = pd.DataFrame(data=conf_matrix,index=names_rows,columns=names_columns)
            
            # compute final evaluation measures
            precision_per_class = np.zeros((self.vw.n_tags,))
            recall_per_class = np.zeros((self.vw.n_tags,))
            
            for i in range(self.vw.n_tags):
                if conf_matrix.values[i,i] > 0:

                    precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
    
                    recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])

                    f1_per_class = 2*(precision_per_class[i]*recall_per_class[i])/(precision_per_class[i]+recall_per_class[i])
                    self.performance.plot_precision_class[i].append(precision_per_class[i])
                    self.performance.plot_recall_class[i].append(recall_per_class[i])
                    self.performance.plot_f1_class[i].append(f1_per_class)
                else:
                    self.performance.plot_precision_class[i].append(0)
                    self.performance.plot_recall_class[i].append(0)
                    self.performance.plot_f1_class[i].append(0)
            precision = np.mean(precision_per_class)
            recall = np.mean(recall_per_class)
            f1 = 2*(precision*recall)/(precision+recall)
            self.performance.plot_precision_class[self.vw.n_tags].append(precision)
            self.performance.plot_recall_class[self.vw.n_tags].append(recall)
            self.performance.plot_f1_class[self.vw.n_tags].append(f1)
            #this eval only works, if we use validation_split = 0.2 as additional parameter for model.fit()

