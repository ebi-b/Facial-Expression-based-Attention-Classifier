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
from sklearn.metrics import  accuracy_score, roc_auc_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import facial_expression_functions as fef
import omron_expressions_functions as oef
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, confusion_matrix, auc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

class FacialExpressionClassifierChal:

    def __init__(self, participants, margin_for_eng, margin_for_challenge):
        #args = sys.argv[1]
        #self.participants = self.load_participants(path_of_participant_objects)
        self.participants = participants
        self.margin_for_engagement = margin_for_eng
        self.margin_for_challenge = margin_for_challenge
        #self.classification_datapoint_based()
        #self.test_parameters()
        self.classification_participant_based()

    def classification_participant_based(self):
        print("in participant based classifier...")
        self.xgboost_classifier_participant_based()
        #self.knn_classifier_participant_based()

    def classification_datapoint_based(self):
        #print("In classification Processing...")
        raw_q_labels = []
        raw_e_labels = []
        raw_c_labels = []
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
                raw_q_labels.append(quadrant_label)
                raw_e_labels.append(eng_label)
                raw_c_labels.append(chal_label)
                if eng_label != "mr":
                    if chal_label !="mr":
                        try:
                            tmp = []
                            au_c_avg, au_c_std, au_r_avg, au_r_std = fef.calculate_action_units_parameters(point, 30)
                            gaze_avg_movements, gaze_avg, gaze_std = fef.calculate_gaze_angle_parameters(point)
                            avg_movement_head_pose, head_pose_avg, head_pose_std = fef.calculate_head_pose(point)
                            avg_movement_pitch_roll_yaw, pitch_roll_yaw_avg, pitch_roll_yaw_std = fef.calculate_pitch_roll_yaw(point)
                            #rint("toole au_c_avg is: ", len(au_c_avg))
                            if sum(au_c_avg) == 0:
                                #print("SUM++00")
                                abas=1
                            if len(au_c_avg) != 0 and sum(au_c_avg) != 0:
                                eng_labels.append(eng_label)
                                chal_labels.append(chal_label)
                                quadrant_labels.append(quadrant_label)
                                tmp.extend(au_r_avg)
                                tmp.extend(au_r_std)
                                tmp.extend(au_c_avg)
                                tmp.extend(au_c_std)

                                tmp.extend(gaze_avg_movements)
                                tmp.extend(gaze_avg)
                                tmp.extend(gaze_std)

                                tmp.extend(avg_movement_head_pose)
                                tmp.extend(head_pose_avg)
                                tmp.extend(head_pose_std)

                                tmp.extend(avg_movement_pitch_roll_yaw)
                                #print("avg: {0}".format(pitch_roll_yaw_avg))
                                tmp.extend(pitch_roll_yaw_avg)
                                #print("std: {0}".format(pitch_roll_yaw_std))
                                tmp.extend(pitch_roll_yaw_std)

                                au_cr.append(tmp)
                        except(TypeError):
                            print("Error in participant {0} and timestamp {1}.".format(point.participant_number, point.rate.timestamp))
        #for i in range(len(au_cr)):
            #print("Len row ", i," is: ",len(au_cr[i]))
        #np_au_cr = numpy.array(au_cr)
        #np_au_cr.reshape((118,70))
        #print(np_au_cr.shape)
        #print(numpy.shape(au_cr))
        #print(numpy.shape(quadrant_labels))
        #print(quadrant_labels)
        numbers_q = np.zeros(5)
        numbers_e = np.zeros(3)
        numbers_c = np.zeros(3)
        for index in raw_q_labels:
            if index == 1:
                numbers_q[0] += 1
            if index == 2:
                numbers_q[1] += 1
            if index == 3:
                numbers_q[2] += 1
            if index == 4:
                numbers_q[3] += 1
            if index == 'mr':
                numbers_q[4] += 1
        avg_q = numbers_q/sum(numbers_q)
        print("avg_q : {0} ".format(avg_q))
        print("labels_q : {0} ".format(numbers_q))

        for i in range(len(raw_e_labels)):
            if raw_e_labels[i] == 0:
                numbers_e[0] += 1
            if raw_e_labels[i] == 1:
                numbers_e[1] += 1
            if raw_e_labels[i] == 'mr':
                numbers_e[2] += 1
            if raw_c_labels[i] == 0:
                numbers_c[0] += 1
            if raw_c_labels[i] == 1:
                numbers_c[1] += 1
            if raw_c_labels[i] == 'mr':
                numbers_c[2] += 1
        #avg_e = numbers_e/sum(numbers_e)
        #print("avg_e : {0} ".format(avg_e))
        #print("labels_e : {0} ".format(numbers_e))
        #avg_c = numbers_c / sum(numbers_c)
        #print("avg_c : {0} ".format(avg_c))
        #print("labels_c : {0} ".format(numbers_c))


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

            X_train,X_test = self.pca_transformation(X_train, X_test)
            SVM_quadrant = svm.LinearSVC(class_weight='balanced', max_iter=1000000)
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
        self.random_forest(au_cr, quadrant_labels)
        self.logistic_regression_classifier(au_cr, quadrant_labels)

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

            X_train, X_test = self.pca_transformation(X_train, X_test)
            SVM = svm.SVC(kernel= 'poly', degree=2, gamma='scale', max_iter=3000, class_weight='balanced')
            SVM.fit(X_train, y_train)
            pt = SVM.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def xgboost_classifier(self,au_cr, quadrant_labels):
        print("XGBOOST")
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

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            smt = SMOTE()
            X_train, y_train = smt.fit_sample(X_train, y_train)
            #SVM = svm.SVC(gamma='scale', class_weight='balanced')
            #SVM.fit(X_train, y_train)
            #pt = SVM.predict(X_test)
            #X_train, X_test = self.pca_transformation(X_train, X_test)
            model = xgb.XGBClassifier(objective='multi:softmax', max_iter=3000).fit(X_train , y_train)
            pt = model.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def random_forest(self,au_cr, quadrant_labels):
        print("random forest")
        # Leave One Out
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

            X_train, X_test = self.pca_transformation(X_train, X_test)
            model = RandomForestClassifier(n_estimators = 100)
            model.fit(X_train, y_train)
            pt =model.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)

        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def pca_transformation(self, X_train, X_test ):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        return X_train, X_test

    @staticmethod
    def load_participants(path):
        participants = []
        for filename in os.listdir(path):
            file_pi2 = open(str(path) + "\\" + str(filename), 'rb')
            participant = pickle.load(file_pi2)
            print("Participant {0} is Loaded.".format(filename))
            if participant.number != 55:
                participants.append(participant)
        # filehandler = open("participants.obj", 'wb')
        # pickle.dump(participants, filehandler)
        return participants

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

        if datapoint.rate.challenge == 0 or datapoint.rate.engagement == 0:
            quadrant_label = 'mr'
            engagement_label = 'mr'
            challenge_label = "mr"

        return quadrant_label, engagement_label, challenge_label
        #print("quad label is: ", self.quadrant_label)

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

    def logistic_regression_classifier(self, au_cr, quadrant_labels):
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

            X_train, X_test = self.pca_transformation(X_train, X_test)
            model = LogisticRegressionCV(class_weight='balanced', multi_class='ovr')
            model.fit(X_train, y_train)
            pt = model.predict(X_test)
            predicted.extend(pt)
            real.extend(y_test)
        print("Logistic Regression is : ")
        print("Y:", real)
        print("Predicted:", predicted)
        accuuracy = accuracy_score(real, predicted)
        print("Accuracy is: ", accuuracy)

    def xgboost_classifier_participant_based(self):
        print("XGBOOST-Participant Based")
        predicted_rf = []
        real_rf = []

        predicted_nb = []
        real_nb = []

        predicted_poly = []
        real_poly = []

        predicted_xgb = []
        real_xgb = []

        predicted_lsvm = []
        real_lsvm = []

        eng_labels = []
        chal_labels = []
        quadrant_labels = []
        au_cr = []
        data_points = []
        train_dps = []
        test_dps = []
        for k in range(len(self.participants)):
            for dp in self.participants[k].data_points:
                if dp.rate.engagement != 0 and dp.rate.challenge != 0:
                    data_points.append(dp)

        au_cr, chal_labels, p_numbers = self.sliding_windows_preparing_features()
        print(len(p_numbers))
        self.plot_pca_2_level(au_cr, chal_labels)
        #self.plot_pca_4_level(au_cr, quadrant_labels)

        #au_cr = normalize(au_cr, axis=0, norm='max')
        #au_cr =StandardScaler
        for i in range(5, 21):
            if (i != 13 and i != 18):
                # for participant in self.participants:
                # participant_number = participant.number
                # print("Participant_number: "+str(participant_number))
                print("iteration " + str(i))
                train_au_cr, train_chal_labels = [], []
                test_au_cr, test_chal_labels = [], []
                for j in range(len(p_numbers)):
                    if p_numbers[j] != i:
                        train_au_cr.append(au_cr[j])
                        train_chal_labels.append(chal_labels[j])
                    else:
                        test_au_cr.append(au_cr[j])
                        test_chal_labels.append(chal_labels[j])


                X_train = train_au_cr
                X_test = test_au_cr
                y_train = train_chal_labels
                y_test = test_chal_labels
                X_test = np.array(X_test)
                scaler=StandardScaler().fit(X_train)
                if (len(X_test)==1):
                    X_test=X_test.reshape(1, -1)
                scaler.transform(X_train)
                scaler.transform(X_test)

                if (len(y_test) != 0):
                    X_train = np.array(X_train)
                    X_test = np.array(X_test)
                    y_train = np.array(y_train)
                    y_test = np.array(y_test)

                    #smt = SMOTE()
                    #X_train, y_train = smt.fit_sample(X_train, y_train)
                    X_train, X_test = self.pca_transformation(X_train, X_test)

                    rf_model = RandomForestClassifier(n_estimators=500)
                    rf_model.fit(X_train, y_train)
                    rf_pt = rf_model.predict(X_test)
                    predicted_rf.extend(rf_pt)
                    real_rf.extend(y_test)

                    nb_model = GaussianNB()
                    nb_model.fit(X_train, y_train)
                    nb_pt = nb_model.predict(X_test)
                    predicted_nb.extend(nb_pt)
                    real_nb.extend(y_test)

                    xgb_model = xgb.XGBClassifier(objective='binary:logistic', max_iter=3000).fit(X_train, y_train)
                    xgb_pt = xgb_model.predict(X_test)
                    predicted_xgb.extend(xgb_pt)
                    real_xgb.extend(y_test)

                    SVM_linear = svm.LinearSVC(class_weight='balanced', max_iter=1000000)
                    SVM_linear.fit(X_train, y_train)
                    pt_lsvm = SVM_linear.predict(X_test)
                    predicted_lsvm.extend(pt_lsvm)
                    real_lsvm.extend(y_test)

                    SVM_poly = svm.SVC(kernel='poly', degree=2, gamma='scale', max_iter=3000, class_weight='balanced')
                    SVM_poly.fit(X_train, y_train)
                    pt_poly = SVM_poly.predict(X_test)
                    predicted_poly.extend(pt_poly)
                    real_poly.extend(y_test)

        print("------------RANDOM FOREST-------")
        print(classification_report(real_rf, predicted_rf))
        print("Other Metrics: \n")
        print(confusion_matrix(real_rf, predicted_rf))
        print("Y:", real_rf)
        print("Predicted:", predicted_rf)
        accuuracy = accuracy_score(real_rf, predicted_rf)
        print("Accuracy is: ", accuuracy)

        print("------------NAIVE BAYES-------")
        print(classification_report(real_nb, predicted_nb))
        print("Other Metrics: \n")
        print(confusion_matrix(real_nb, predicted_nb))
        print("Y:", real_nb)
        print("Predicted:", predicted_nb)
        accuuracy = accuracy_score(real_nb, predicted_nb)
        print("Accuracy is: ", accuuracy)

        print("------------ XGB -------")
        print(classification_report(real_xgb, predicted_xgb))
        print("Other Metrics: \n")
        print(confusion_matrix(real_xgb, predicted_xgb))
        print("Y:", real_xgb)
        print("Predicted:", predicted_xgb)
        accuuracy = accuracy_score(real_xgb, predicted_xgb)
        print("Accuracy is: ", accuuracy)

        print("------------ LSVM -------")
        print(classification_report(real_lsvm, predicted_lsvm))
        print("Other Metrics: \n")
        print(confusion_matrix(real_lsvm, predicted_lsvm))
        print("Y:", real_lsvm)
        print("Predicted:", predicted_lsvm)
        accuuracy = accuracy_score(real_lsvm, predicted_lsvm)
        print("Accuracy is: ", accuuracy)

        print("------------ Poly SVM -------")
        print(classification_report(real_poly, predicted_poly))
        print("Other Metrics: \n")
        print(confusion_matrix(real_poly, predicted_poly))
        print("Y:", real_poly)
        print("Predicted:", predicted_poly)
        accuuracy = accuracy_score(real_poly, predicted_poly)
        print("Accuracy is: ", accuuracy)

    def knn_classifier_participant_based(self):
        print("KNN-Participant Based")
        predicted = []
        real = []
        eng_labels = []
        chal_labels = []
        quadrant_labels = []
        au_cr = []
        data_points = []
        train_dps = []
        test_dps = []
        for k in range(len(self.participants)):
            predicted = []
            real = []
            for dp in self.participants[k].data_points:
                data_points.append(dp)

        au_cr, chal_labels, p_numbers = self.prepare_features_for_challenge(data_points)

        for number_of_neighbors in range(1,11):
            for i in range(5 , 21):
                train_au_cr, train_chal_labels = [], []
                test_au_cr, test_chal_labels = [], []
                for j in range(len(p_numbers)):
                    if p_numbers[j] != i:
                            train_au_cr.append(au_cr[j])
                            train_chal_labels.append(chal_labels[j])

                    else:
                            test_au_cr.append(au_cr[j])
                            test_chal_labels.append(chal_labels[j])

                X_train = train_au_cr
                X_test = test_au_cr
                y_train = train_chal_labels
                y_test = test_chal_labels
                #print("here")
                if(len(y_test) != 0):
                    X_train = np.array(X_train)
                    X_test = np.array(X_test)
                    y_train = np.array(y_train)
                    y_test = np.array(y_test)
                    X_train, X_test = self.pca_transformation(X_train, X_test)
                    smt = SMOTE()
                    X_train, y_train = smt.fit_sample(X_train, y_train)
                    #SVM = svm.SVC(gamma='scale', class_weight='balanced')
                    #SVM.fit(X_train, y_train)
                    #pt = SVM.predict(X_test)
                    X_train, X_test = self.pca_transformation(X_train, X_test)
                    #model = xgb.XGBClassifier(objective='multi:softmax', max_iter=3000).fit(X_train , y_train)
                    model = KNeighborsClassifier(n_neighbors=number_of_neighbors)
                    model.fit(X_train, y_train)
                    pt = model.predict(X_test)
                    predicted.extend(pt)
                    real.extend(y_test)
                    mid_accuuracy = accuracy_score(y_test, pt)
                    print("Mid-accuaracy is {0}, y_test is {1} ".format(mid_accuuracy, y_test))

            #print("Y:", real)
            #print("Predicted:", predicted)
            accuuracy = accuracy_score(real, predicted)
            #auc_score = roc_auc_score(real, predicted)
            #print("auc_score is: ", auc_score)
            print("Accuracy is: ", accuuracy)
            print(" K is : "+str(number_of_neighbors))
            print(classification_report(real, predicted))
            print("Other Metrics: \n")
            print(confusion_matrix(real, predicted))

    def prepare_features_for_challenge(self, data_points):
        chal_labels, au_cr, p_numbers = [], [], []
        if type(data_points) != 'list':
            data_points = [data_points]
        for i in range(len(data_points)):
            point = data_points[i]
            # for point in data_points[i]:
            quadrant_label, eng_label, chal_label = self.calculate_labels(point)
            if chal_label != "mr":
                    try:
                        tmp = []
                        au_c_avg, au_c_std, au_r_avg, au_r_std = fef.calculate_action_units_parameters(point, 30)
                        gaze_avg_movements, gaze_avg, gaze_std = fef.calculate_gaze_angle_parameters(point)
                        avg_movement_head_pose, head_pose_avg, head_pose_std = fef.calculate_head_pose(point)
                        avg_movement_pitch_roll_yaw, pitch_roll_yaw_avg, pitch_roll_yaw_std = fef.calculate_pitch_roll_yaw(
                            point)
                        avg_sadness_rate, std_sadness_rate, median_sadness_rate, avg_anger_rate, std_anger_rate, median_anger_rate \
                            , avg_surprise_rate, std_surprise_rate, median_surprise_rate, avg_happines_rate, std_happiness_rate, \
                        median_happiness_rate, avg_neutral_rate, std_neutral_rate, median_neutral_rate = \
                            oef.calculate_omron_emotions_score_array_metric(point)

                        avg_exp, max_a = oef.calculate_omron_expression_array_metric(point)

                        if sum(au_c_avg) == 0:
                            # print("SUM++00")
                            abas = 1
                        if len(au_c_avg) != 0 and sum(au_c_avg) != 0:
                            chal_labels.append(chal_label)
                            p_numbers.append(point.participant_number)
                            tmp.extend(au_r_avg)
                            tmp.extend(au_r_std)
                            tmp.extend(au_c_avg)
                            tmp.extend(au_c_std)

                            tmp.extend(gaze_avg_movements)
                            tmp.extend(gaze_avg)
                            tmp.extend(gaze_std)

                            tmp.extend(avg_movement_head_pose)
                            tmp.extend(head_pose_avg)
                            tmp.extend(head_pose_std)

                            tmp.extend(avg_movement_pitch_roll_yaw)
                            # print("avg: {0}".format(pitch_roll_yaw_avg))
                            tmp.extend(pitch_roll_yaw_avg)
                            # print("std: {0}".format(pitch_roll_yaw_std))
                            tmp.extend(pitch_roll_yaw_std)

                            # OMRON FEATURES::

                            omron_emotion_array = [avg_sadness_rate, std_sadness_rate, median_sadness_rate,
                                                   avg_anger_rate, std_anger_rate, median_anger_rate, avg_surprise_rate,
                                                   std_surprise_rate, median_surprise_rate, avg_happines_rate,
                                                   std_happiness_rate, median_happiness_rate, avg_neutral_rate,
                                                   std_neutral_rate, median_neutral_rate]

                            tmp.extend(omron_emotion_array)

                            tmp.extend(avg_exp)
                            tmp.extend(max_a)
                            # print("Len TMP is: "+str(len(tmp)))

                            au_cr.append(tmp)
                    except(TypeError):
                        print(TypeError)
                        print("Error in participant {0} and timestamp {1}.".format(point.participant_number,
                                                                                   point.rate.timestamp))
        return au_cr,chal_labels, p_numbers


    def sliding_windows_preparing_features(self):
        p_au_cr, p_quadrant_labels, p_eng_labels, p_chal_labels, p_p_numbers = [],[],[],[],[]
        not_set_yet = True
        for i in range(len(self.participants)):
            for j in range(i, len(self.participants)):
                if i != j:
                    if self.participants[i].number == self.participants[j].number:
                        au_cr, chal_labels, p_numbers = \
                            self.sliding_windows_mixing_datapoints(self.participants[i], self.participants[j])
                        au_cr = numpy.array(au_cr)

                        #quadrant_labels = numpy.array(quadrant_labels)
                        #eng_labels = numpy.array(eng_labels)
                        chal_labels = numpy.array(chal_labels)
                        p_numbers = numpy.array(p_numbers)
                        if not_set_yet:
                            if(len(au_cr) > 0):
                                p_au_cr = au_cr
                                #p_quadrant_labels = quadrant_labels
                                #p_eng_labels = eng_labels
                                p_chal_labels = chal_labels
                                p_p_numbers = p_numbers
                                not_set_yet = False
                        else:
                            if(len(au_cr) > 0):
                                p_au_cr = np.concatenate((p_au_cr, au_cr), axis=0)
                                #p_quadrant_labels = np.concatenate((p_quadrant_labels, quadrant_labels))
                                #p_eng_labels = np.concatenate((p_eng_labels, eng_labels))
                                p_chal_labels = np.concatenate((p_chal_labels, chal_labels))
                                p_p_numbers = np.concatenate((p_p_numbers, p_numbers))
        return p_au_cr, p_chal_labels, p_p_numbers

    def sliding_windows_mixing_datapoints(self, participant_1,participant_2):
        au_cr,quadrant_labels, eng_labels, chal_labels, p_numbers = [],[],[],[],[]
        for i in range(len(participant_1.data_points)):
            for j in range(len(participant_2.data_points)):
                if participant_1.data_points[i].rate.timestamp == participant_2.data_points[j].rate.timestamp:

                    au_cr_1,  chal_label, p_number = self.prepare_features_for_challenge(participant_1.data_points[i])
                    au_cr_2,  chal_label, p_number = self.prepare_features_for_challenge(participant_2.data_points[j])
                    if len(au_cr_1) > 0 and len(au_cr_2) > 0:
                        au_cr_2 = np.array(au_cr_2)
                        au_cr_1 = np.array(au_cr_1)
                        au_cr_2 = au_cr_2.flatten()
                        au_cr_1 = au_cr_1.flatten()
                        tmp_au_cr = [*au_cr_1, *au_cr_2]
                        tmp_au_cr = np.array(tmp_au_cr)
                        #tmp_au_cr = tmp_au_cr.flatten()
                        au_cr.append(tmp_au_cr)

                        chal_labels.append(chal_label)
                        p_numbers.append(p_number)
        return au_cr, chal_labels, p_numbers

    def plot_pca_2_level(self, au_cr, labels):
        au_cr, x =self.pca_transformation(au_cr, au_cr)
        h = []
        l = []
        for i in range(len(labels)):
            if(labels[i] == 0):
                l.append(au_cr[i])
            if (labels[i] == 1):
                h.append(au_cr[i])
        h = numpy.array(h)
        l = numpy.array(l)
        plt.scatter(l[:,0], l[:,1], c = "red")
        plt.scatter(h[:,0], h[:,1], c = "green")
        plt.show()

