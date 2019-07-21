import array
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from openface_object import Openface
import os
from decimal import Decimal
from rates import Rate
from interruption import Interruptions
from data_point import DataPoint
import rarfile

class Participant:

    def __init__(self, number, date, age, gender, path_of_logs):
        self.number = number
        self.date = date
        self.age = age
        self.gender = gender
        self.rates = []
        self.interruptions = []
        self.data_points = []
        self.set_rates(path_of_logs)
        self.path_of_participant_snapshots = ""
        self.generate_data_points()

    # This function is for setting path of snapshots
    def set_path_of_participant_snapshots(self, path_of_snapshots):
        self.path_of_participant_snapshots = path_of_snapshots

    # This function set user engagement and challenge inputs in Participant Object
    def set_rates(self, path):
        lastInteruptionTimeStamp=""
        lastChallengeTimeStamp=""
        lastEngagementTimeStamp=""
        lastEngagementValue=0
        lastChallengeValue=0
        lastSubmitTimeStamp=""
        lastInteruptionTimeStamp=""
        print("Setting Rates for Participant {0}.".format(self.number))
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                #tmp1 = "Sliders CSVs/Engagement and Challenge/" + filename + "E.csv"
                #Pallete_pp_CSV = open(tmp1, "a")
                #tmp2 = "Sliders CSVs/Interuptions/" + filename + "I.csv"
                #Interuption_pp_CSV = open(tmp2, "a")
                #sliders_pp_writer = csv.writer(Pallete_pp_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #interruption_pp_writer = csv.writer(Interuption_pp_CSV, delimiter=',')
                sliderLogFile = open(path + '/' + filename, 'r')
                lenght_of_file = len(open(path + '/' + filename).readlines())

                lastChallengeTimeStamp = '0'
                lastChallengeValue = 0
                lastSubmitTimeStamp = '0'
                lastInteruptionTimeStamp = '0'
                for i in range(lenght_of_file):
                    line = sliderLogFile.readline()
                    # print(str(i) +' : ' +line)
                    if line.find("Challenge") != -1:
                        try:
                            lastChallengeStringValue = Participant.find_value(line)
                            lastChallengeValue = Participant.map_values(lastChallengeStringValue, 'c')
                            lastChallengeTimeStamp = Participant.find_time_stamp(line)
                        except:
                            print("Exception: " + line)
                    if line.find("Engagement") != -1:
                        try:
                            lastEngagementStringValue = Participant.find_value(line)
                            lastEngagementValue = Participant.map_values(lastEngagementStringValue, 'e')
                            lastEngagementTimeStamp = Participant.find_time_stamp(line)
                        except:
                            print("Exception: " + line)

                    if line.find("Submit") != -1:
                        #print(line)
                        lastSubmitTimeStamp = Participant.find_time_stamp(line)
                        tmp_rate = Rate(lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp,
                                        lastChallengeValue, lastChallengeTimeStamp)
                        self.rates.append(tmp_rate)

                        #sliders_writer.writerow(
                        #   [lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp, lastChallengeValue,
                        #    lastChallengeTimeStamp])
                        #sliders_pp_writer.writerow(
                        #   [lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp, lastChallengeValue,
                        #    lastChallengeTimeStamp])
                        # writeString=lastSubmitTimeStamp+','+str(lastEngagementValue)+','+lastEngagementTimeStamp+','+str(lastChallengeValue)+','+lastChallengeTimeStamp+';'
                        # PalleteCSV.write(writeString)

                    if line.find("Interuption") != -1:
                        lastInteruptionTimeStamp = Participant.find_time_stamp(line)
                        tmp_interruption= Interruptions(lastInteruptionTimeStamp,'--Reason--')
                        #interruption_writer.writerow([lastInteruptionTimeStamp, '--Reason-'])
                        #interruption_pp_writer.writerow([lastInteruptionTimeStamp, '--Reason-'])
                        # writeString=lastInteruptionTimeStamp+',--;'
                        # InteruptionCSV.write(writeString)

                continue
            else:
                continue

    def generate_data_points(self):
        for rate in self.rates:
            tmp_data_point = DataPoint(self.number, rate)
            self.data_points.append(tmp_data_point)

    def set_data_point(self, datapoins):
        self.data_points = datapoins

    # This function find values of engagement or challenge in a line of pallete log.
    @staticmethod
    def find_value(s):
        j1 = s.split('{')
        j2 = j1[1].split('[')
        j3 = j2[1].split(']')
        value = j3[0]
        return value

    # This function find timestamp in a line of pallete log.
    @staticmethod
    def find_time_stamp(s):
        j = s.split("{")
        d = j[0].split(" ")
        time = d[2]
        decimal_time = Decimal(time)
        decimal_time = decimal_time / 1000
        return str(decimal_time)

    # This function map the values of logs to real values.
    @staticmethod
    def map_values(sv, t):
        v = float(sv)
        minIndex = 0
        cLevel = array.array('f',
                             [0, 0.39, 5.1, 10.98, 17.25, 21.18, 26.27, 32.55, 38.04, 43.53, 48.24, 53.73, 60, 64.71,
                              69.41, 75.69, 82.35, 87.06, 92.55, 99.22, 99.61])
        eLevel = array.array('f', [0, 0.39, 3.14, 9.02, 15.29, 20, 25.1, 30.2, 36.47, 41.57, 46.67, 52.55, 57.65, 63.53,
                                   69.02, 74.12, 80.78, 85.1, 91.37, 99.22, 99.61])
        if t == 'c':
            mindif = 1000
            minIndex = 0
            for i in range(0, 21):
                dif = abs(cLevel[i] - v)
                if dif < mindif:
                    minIndex = i
                    mindif = dif

        if t == 'e':
            mindif = 1000
            minIndex = 0
            for i in range(0, 21):
                # print(i)
                dif = abs(eLevel[i] - v)
                if dif < mindif:
                    minIndex = i
                    mindif = dif

        realValue = minIndex * 5
        return realValue

    # This function find related frames [based on period and margin] to each data point and copy them in
    # [self.path_for_saving_datapoint_frames ] for further analysis and return participants with setted openface object[including CSVs and all features set]
    def preparation_for_facial_expression_analysis(self, period, margin, path_for_saving_datapoint_frames, ):
        snapshot_files_name = []
        refinement = True
        set_before = False
        if not refinement:
            if len(listdir(self.path_of_participant_snapshots)) == 0:
                print("ALERT: Directory " + self.path_of_participant_snapshots+ " is EMPTY.")

        else:
            for f in listdir(self.path_of_participant_snapshots):
                if isfile(join(self.path_of_participant_snapshots, f)):
                    snapshot_files_name.append(f)

            data_points_with_openface = []
            for datapoint in self.data_points:
                rate = datapoint.rate
                print("Point: " + str(rate.timestamp))
                number_of_snapshots = 0
                start_time_stamp = rate.timestamp - period
                end_time_stamp = rate.timestamp - margin
                folder_path = ""
                try:
                    folder_path = str(path_for_saving_datapoint_frames + "\\" + str(self.number) + "\\"
                                      + str(rate.timestamp))
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    else: set_before = True
                    #datapoint..set_webcam_snapshots_for_path(folder_path)

                except OSError:
                    print("Creation of the directory %s failed" % folder_path)
                    print(OSError.strerror())
                else:
                    print("Successfully created the directory %s " % folder_path)
                # Here we copy snapshots in destination directories
                snapshot_files_timestamp = []
                if not set_before:
                    for name in snapshot_files_name:
                        try:
                            tmp = name.replace(".jpg", "")
                            snapshot_files_timestamp.append(float(tmp))
                            #print("File: " + tmp)
                        except:
                            print("")

                    for t in snapshot_files_timestamp:
                        if start_time_stamp < t < end_time_stamp:
                            src = str(self.path_of_participant_snapshots + "\\" + str(t) + ".jpg")
                            dst = str(folder_path + "\\" + str(t) + ".jpg")
                            copyfile(src, dst)
                            number_of_snapshots = number_of_snapshots + 1

                open_face_object = Openface(datapoint.rate, folder_path, datapoint.participant_number)
                datapoint.set_openface_object(open_face_object)
                # rate.set_number_of_snapshots(number_of_snapshots)
                data_points_with_openface.append(datapoint)

            self.data_points = data_points_with_openface

    def preparation_for_facial_expression_analysis_with_rar_file(self, period, margin, path_for_saving_datapoint_frames):
        snapshot_files_name = []
        if False:
            print("ALERT: Directory " + self.path_of_participant_snapshots+ " is EMPTY.")
        else:
            # Set to full path of unrar.exe if it is not in PATH
            rarfile.UNRAR_TOOL = "C:\\Program Files\\WinRAR\\UnRAR.exe"
            # Set to '\\' to be more compatible with old rarfile
            rarfile.PATH_SEP = '/'

            snapshot_files_name = []
            path_of_all_snapshots = self.path_of_participant_snapshots
            snapshot_files_timestamp = []
            rf = rarfile.RarFile(path_of_all_snapshots)
            folder_in_rar = ""
            for f in rf.infolist():
                if not f.isdir():
                    #print(f.filename)
                    s = f.filename.split('/')
                    folder_in_rar = s[0]
                    name = s[1].replace(".jpg", "")
                    snapshot_files_name.append(f.filename)
                    snapshot_files_timestamp.append(name)
            #print(snapshot_files_name)

            data_points_with_openface = []
            for datapoint in self.data_points:
                rate = datapoint.rate
                print("Point: " + str(rate.timestamp))
                number_of_snapshots = 0
                start_time_stamp = rate.timestamp - period
                end_time_stamp = rate.timestamp - margin
                folder_path = ""
                try:
                    folder_path = str(path_for_saving_datapoint_frames + "\\" + str(self.number) + "\\"
                                      + str(rate.timestamp))
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)


                    #datapoint..set_webcam_snapshots_for_path(folder_path)

                except OSError:
                    print("Creation of the directory %s failed" % folder_path)
                    print(OSError.strerror())
                else:
                    print("Successfully created the directory %s " % folder_path)
                # Here we copy snapshots in destination directories
                for t in range(len(snapshot_files_timestamp)):
                    if start_time_stamp < float(snapshot_files_timestamp[t]) < end_time_stamp:
                        src = str(snapshot_files_name[t])
                        dst = str(folder_path)
                        rf.extract(src, dst)
                        number_of_snapshots = number_of_snapshots + 1

                os.rename(str(folder_path + "\\" + folder_in_rar), str(folder_path + "\\" + str(datapoint.rate.timestamp)))

                path_in_rar = folder_path+"\\"+str(datapoint.rate.timestamp)
                open_face_object = Openface(datapoint.rate, path_in_rar, datapoint.participant_number)
                datapoint.set_openface_object(open_face_object)
                # rate.set_number_of_snapshots(number_of_snapshots)
                data_points_with_openface.append(datapoint)

            self.data_points = data_points_with_openface

    def split_to_mini_datapoints(self,lenght):
        mini_data_points = []
        for datapoint in self.data_points:
            mini_data_points.append(datapoint.to_mini_data_points(lenght))
        new_mini_data_points = []
        for i in range(len(mini_data_points)):
            for j in range(len(mini_data_points[i])):
                new_mini_data_points.append(mini_data_points[i][j])
        return new_mini_data_points

