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
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import facial_expression_functions as fef

class FacialExpressionClassifier:

    def __init__(self, path_of_participant_objects, margin_for_eng, margin_for_challenge):
        self.participants = self.load_participants(path_of_participant_objects)
        self.margin_for_engagement = margin_for_eng
        self.margin_for_challenge = margin_for_challenge
        #self.classification_datapoint_based()
        self.test_parameters()


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
                            au_c_avg, au_c_std, au_r_avg, au_r_std = fef.calculate_action_units_parameters (point, 195)
                            # gaze_avg_movements, gaze_avg, gaze_std
                            print("toole au_c_avg is: ", len(au_c_avg))
                            if sum(au_c_avg) == 0:
                                print("SUM++00")
                            if len(au_c_avg) != 0 and sum(au_c_avg) != 0:
                                eng_labels.append(eng_label)
                                chal_labels.append(chal_label)
                                quadrant_labels.append(quadrant_label)
                                tmp.extend(au_r_avg)
                                tmp.extend(au_r_std)
                                tmp.extend(au_c_avg)
                                tmp.extend(au_c_std)
                                #tmp.extend(gaze_avg_movements)
                                #tmp.extend(gaze_avg)
                                #tmp.extend(gaze_std)
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


        predicted_quadran = []
        real_quadrant = []

        predicted_eng = []
        real_eng = []

        predicted_chal = []
        real_chal = []

        for train_index, test_index in loo.split(au_cr):
        #for train_index, test_index in kf.split(au_cr):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_test = []
            X_train = []
            quadrant_y_test = []
            eng_y_test = []
            chal_y_test = []
            quadrant_y_train = []
            eng_y_train = []
            chal_y_train = []

            for j in train_index:
                X_train.append(au_cr[j])
                quadrant_y_train.append(quadrant_labels[j])
                eng_y_train.append(eng_labels[j])
                chal_y_train.append(chal_labels[j])
            for k in test_index:
                X_test.append(au_cr[k])
                quadrant_y_test.append(quadrant_labels[k])
                eng_y_test.append(eng_labels[k])
                chal_y_test.append(chal_labels[k])

            SVM_quadrant = svm.LinearSVC(class_weight='balanced')
            SVM_quadrant.fit(X_train, quadrant_y_train)
            pt = SVM_quadrant.predict(X_test)
            predicted_quadran.extend(pt)
            real_quadrant.extend(quadrant_y_test)

            SVM_eng = svm.LinearSVC()
            SVM_eng.fit(X_train, eng_y_train)
            pt = SVM_eng.predict(X_test)
            predicted_eng.extend(pt)
            real_eng.extend(eng_y_test)

            SVM_chal = svm.LinearSVC()
            SVM_chal.fit(X_train, chal_y_train)
            pt = SVM_chal.predict(X_test)
            predicted_chal.extend(pt)
            real_chal.extend(chal_y_test)

        print("Linear")
        print("Y:", real_quadrant)
        print("Predicted:", predicted_quadran)
        accuuracy_quadrant = accuracy_score(real_quadrant, predicted_quadran)
        accuuracy_eng = accuracy_score(real_eng, predicted_eng)
        accuuracy_chal= accuracy_score(real_chal, predicted_chal)
        #print(precision_recall_fscore_support(real_quadrant, predicted_quadran))
        print("Accuracy quadrant is: ", accuuracy_quadrant)
        print("Accuracy eng is: ", accuuracy_eng)
        print("Accuracy chal is: ", accuuracy_chal)

        self.kernel_classifier(au_cr, quadrant_labels)
        self.xgboost_classifier(au_cr, quadrant_labels)

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

    @staticmethod
    def load_participants(path):
        participants = []
        for filename in os.listdir(path):
            file_pi2 = open(str(path) + "\\" + str(filename), 'rb')
            participant = pickle.load(file_pi2)
            print("Participant {0} is Loaded.".format(filename))
            participants.append(participant)
        #filehandler = open("participants.obj", 'wb')
        #pickle.dump(participants, filehandler)
        return participants

    def kernel_classifier(self, au_cr, quadrant_labels):
        print("Kernel Classifier")
        # Leave One Out
        loo = LeaveOneOut()
        loo.get_n_splits(au_cr)
        # print(loo)

        # k-fold
        kf = KFold(n_splits = 10)
        kf.get_n_splits(au_cr)

        predicted = []
        real = []
        #for train_index, test_index in loo.split(au_cr):
        for train_index, test_index in loo.split(au_cr):
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

            SVM = svm.SVC(gamma='scale', class_weight='balanced')
            SVM.fit(X_train, y_train)
            pt = SVM.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def xgboost_classifier(self,au_cr, quadrant_labels):

        data_dmatrix = xgb.DMatrix(data=au_cr, label=quadrant_labels)
        loo = LeaveOneOut()
        loo.get_n_splits(au_cr)
        # print(loo)

        # k-fold
        kf = KFold(n_splits=10)
        kf.get_n_splits(au_cr)

        predicted = []
        real = []
        # for train_index, test_index in loo.split(au_cr):
        for train_index, test_index in kf.split(au_cr):
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

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            #SVM = svm.SVC(gamma='scale', class_weight='balanced')
            #SVM.fit(X_train, y_train)
            #pt = SVM.predict(X_test)
            model = xgb.XGBClassifier(objective='multi:softmax').fit(X_train , y_train)
            pt = model.predict(X_test)

            predicted.extend(pt)
            real.extend(y_test)

        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def test_parameters(self):
        data_points = []
        for participant in self.participants:
            data_points.append(participant.data_points)
        for i in range(len(data_points)):
            for point in data_points[i]:
                        try:
                            fef.calcualate_facial_expression_parameters(point, 195)
                        except(TypeError):
                            print("Error in participant {0} and timestamp {1}.".format(point.participant_number, point.rate.timestamp))
