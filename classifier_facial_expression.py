import csv
import os
import pickle
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
import random
import sklearn as sk
import pandas as ps
import numpy
from participant import Participant
import matplotlib.cm as cm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import average_precision_score
from sklearn.metrics import  accuracy_score
import numpy as np


class FacialExpressionClassifier:

    def __init__(self, path_of_participant_objects, margin_for_eng, margin_for_challenge):
        self.participants = self.load_participants(path_of_participant_objects)
        self.margin_for_engagement = margin_for_eng
        self.margin_for_challenge = margin_for_challenge
        self.classification_datapoint_based()


    def classification_datapoint_based(self):
        print("In classification Processing...")
        eng_labels = []
        chal_labels = []
        quadrant_labels = []
        au_cr = []
        data_points = []
        for participant in self.participants:
            data_points.append(participant.data_points)
        for i in range(len(data_points)):
            for point in data_points[i]:
                quadrant_label, eng_label, chal_label = self.calculate_labels(point)
                if eng_label != "mr":
                    if chal_label !="mr":
                        try:
                            tmp = []
                            au_c_avg, au_c_std, au_r_avg, au_r_std = self.calculate_metrics(point)
                            print("toole au_c_avg is: ", len(au_c_avg))
                            if len(au_c_avg) != 0:
                                eng_labels.append(eng_label)
                                chal_labels.append(chal_label)
                                quadrant_labels.append(quadrant_label)
                                tmp.extend(au_r_avg)
                                tmp.extend(au_r_std)
                                tmp.extend(au_c_avg)
                                tmp.extend(au_c_std)
                                au_cr.append(tmp)
                        except(TypeError):
                            print("Error in participant {0} and timestamp {1}.".format(point.participant_number, point.rate.timestamp))
        for i in range(len(au_cr)):
            print("Len row ", i," is: ",len(au_cr[i]))
        #np_au_cr = numpy.array(au_cr)
        #np_au_cr.reshape((118,70))
        #print(np_au_cr.shape)
        #print(numpy.shape(au_cr))
        #print(numpy.shape(quadrant_labels))
        #print(quadrant_labels)


        #Leave One Out
        loo = LeaveOneOut()
        loo.get_n_splits(au_cr)
        #print(loo)

        #k-fold
        #kf = KFold(n_splits = 10)
        #kf.get_n_splits(au_cr)


        predicted = []
        real = []
        for train_index, test_index in loo.split(au_cr):
        #for train_index, test_index in kf.split(au_cr):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_test = []
            X_train = []
            y_test = []
            y_train = []
            for j in train_index:
                X_train.append(au_cr[j])
                y_train.append(quadrant_labels[j])
            for k in test_index:
                X_test.append(au_cr[k])
                y_test.append(quadrant_labels[k])

            SVM = svm.LinearSVC()
            SVM.fit(X_train, y_train)
            pt=SVM.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Linear")
        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy=accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)



    def calculate_labels(self, datapoint):
        #print("In set labels...")

        #           |
        #           |
        #    Ro=1   |   Fo=2
        # ----------|----------
        #           |
        #      Bo=3 |   Fr=4
        #           |


        if datapoint.rate.challenge> 50+self.margin_for_challenge:
            challenge_label = 1
            if datapoint.rate.engagement > 50+self.margin_for_engagement:
                quadrant_label = 2
                engagement_label = 1
            elif datapoint.rate.engagement < 50-self.margin_for_engagement:
                quadrant_label = 4
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'

        elif datapoint.rate.challenge < 50-self.margin_for_challenge:
            challenge_label = 0
            if datapoint.rate.engagement > 50+self.margin_for_engagement:
                quadrant_label = 1
                engagement_label = 1
            elif datapoint.rate.engagement < 50-self.margin_for_engagement:
                quadrant_label = 3
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'
        else:
            quadrant_label = "mr"
            challenge_label = 'mr'
            if datapoint.rate.engagement > 50+self.margin_for_engagement:
                engagement_label = 1
            elif datapoint.rate.engagement < 50-self.margin_for_engagement:
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'

        return quadrant_label, engagement_label, challenge_label
        #print("quad label is: ", self.quadrant_label)

    def calculate_metrics(self, point):
        print("Calculating Metrics in processed_data_points for participant {0} and datapoint {1}...".format(point.participant_number, point.rate.timestamp))
        new_array_c = []
        new_array_r = []
        au_c_avg, au_c_std, au_r_avg, au_r_std = [], [], [], []
        if hasattr(point.openface_object, 'au_c_array'):
            for row in point.openface_object.au_c_array:
                if (sum(row) != 0):
                        #print("Row is: ",len(row))
                        #print("New array is: ",len(new_array_c))
                        #np.append(new_array_c,[row], axis=0)
                    new_array_c.append(row)

                #print(new_array_c.s)
            new_array_c = np.array(new_array_c)
            if len(new_array_c) == 0:
                au_c_avg = np.zeros(18)
                au_c_std = np.zeros(18)
                #print("au_c for point {0} of participant {1} is {3}: ".format(point.rate.timestamp,point.participant_number,new_array_c))
            else:
                au_c_avg = new_array_c.mean(axis=0)
                au_c_std = new_array_c.std(axis=0)


            for row in point.openface_object.au_r_array:
                if (sum(row) != 0):
                        #np.append(new_array_r,[row], axis=0)
                    new_array_r.append(row)
            if len(new_array_c) == 0:
                au_r_avg = np.zeros(17)
                au_r_std = np.zeros(17)
                    # print("au_c for point {0} of participant {1} is {3}: ".format(point.rate.timestamp,point.participant_number,new_array_c))
            else:

                new_array_r = np.array(new_array_r)
                au_r_avg = new_array_r.mean(axis=0)
                au_r_std = new_array_r.std(axis=0)

                #print("au_c_avg: ", self.au_c_avg)
                #print("au_r_avg: ", self.au_r_avg)
                #print("au_c_std: ", self.au_c_std)
                #print("au_r_std: ", self.au_r_std)
            #except:
             #   print("Error in participant {0} and datapoint {1}".format(point.participant_number, point.rate.timestamp))
            return au_c_avg, au_c_std, au_r_avg, au_r_std

        else:
            return [], [], [], []
    @staticmethod
    def load_participants(path):
        participants = []
        for filename in os.listdir(path):
            file_pi2 = open(str(path) + "\\" + str(filename), 'rb')
            participant = pickle.load(file_pi2)
            print("Participant {0} is Loaded.".format(filename))
            participants.append(participant)
        return participants

    def kernel_classifier(self, au_cr, quadrant_labels):
        # Leave One Out
        loo = LeaveOneOut()
        loo.get_n_splits(au_cr)
        # print(loo)

        # k-fold
        # kf = KFold(n_splits = 10)
        # kf.get_n_splits(au_cr)

        predicted = []
        real = []
        for train_index, test_index in loo.split(au_cr):
            # for train_index, test_index in kf.split(au_cr):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_test = []
            X_train = []
            y_test = []
            y_train = []
            for j in train_index:
                X_train.append(au_cr[j])
                y_train.append(quadrant_labels[j])
            for k in test_index:
                X_test.append(au_cr[k])
                y_test.append(quadrant_labels[k])

            SVM = svm.SVC(gamma='auto')
            SVM.fit(X_train, y_train)
            pt = SVM.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Linear")
        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)
