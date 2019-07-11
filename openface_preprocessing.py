from participant import Participant
import os
import csv
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import pickle
from openface_object import Openface
import rarfile


class Openface_Preprocessing:

    def __init__(self, participant, period, margin):
        self.participant = Participant(participant)
        self.period = period
        self.path_of_all_snapshots = participant.path_of_snapshots
        self.margin = margin
        self.path_for_saving_snapshots = ""
        self.path_for_saving_datapoint_frames = "Y:\\Openface Processed Frames\\Folders of Datapoints"


    # This function find related frames [based on period and margin] to each data point and copy them in
    # [self.path_for_saving_datapoint_frames ] for further analysis and return participants with setted openface object

    def preparation_for_analysis(self):
        snapshot_files_name = []
        for f in listdir(self.path_of_all_snapshots):
            if isfile(join(self.path_of_all_snapshots, f)):
                snapshot_files_name.append(f)

        data_points_with_openface = []
        for datapoint in self.participant.data_points:
            rate = datapoint.rate
            print("Point: " + str(rate.timestamp))
            number_of_snapshots = 0
            start_time_stamp = rate.timestamp - self.period
            end_time_stamp = rate.timestamp - self.margin
            try:
                folder_path = str(self.path_for_saving_datapoint_frames + "\\" + str(self.participant.number) + "\\"
                                  + str(rate.timestamp))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                self.set_webcam_snapshots_for_path(folder_path)

            except OSError:
                print("Creation of the directory %s failed" % folder_path)
                print(OSError.strerror())
            else:
                print("Successfully created the directory %s " % folder_path)
            # Here we copy snapshots in destination directories
            snapshot_files_timestamp = []
            for name in snapshot_files_name:
                tmp = name.replace(".jpg", "")
                snapshot_files_timestamp.append(float(tmp))
                print("File: " + tmp)

            for t in snapshot_files_timestamp:
                if start_time_stamp < t < end_time_stamp:
                    src = str(self.path_of_all_snapshots + "\\" + str(t) + ".jpg")
                    dst = str(folder_path + "\\" + str(t) + ".jpg")
                    copyfile(src, dst)
                    number_of_snapshots = number_of_snapshots + 1

            open_face_object = Openface(folder_path)
            datapoint.set_openface_object(open_face_object)
            #rate.set_number_of_snapshots(number_of_snapshots)
            data_points_with_openface.append(datapoint)


        participant = self.participant
        participant.set_data_points(data_points_with_openface)
        return participant

    def preparation_for_analysis_with_rar_file(self):
        # Set to full path of unrar.exe if it is not in PATH
        rarfile.UNRAR_TOOL = "C:\\Program Files\\WinRAR\\UnRAR.exe"
        # Set to '\\' to be more compatible with old rarfile
        rarfile.PATH_SEP = '/'

        snapshot_files_name = []
        path_of_all_snapshots = "C:\Webcam Snapshots\webcamSnapshot-8.rar"
        snapshot_files_timestamp = []
        rf = rarfile.RarFile(path_of_all_snapshots)
        for f in rf.infolist():
            if not f.isdir():
                print(f.filename)
                s = f.filename.split('/')
                name = s[1].replace(".jpg", "")
                snapshot_files_name.append(f.filename)
                snapshot_files_timestamp.append(name)
        print(snapshot_files_name)

        #for f in listdir(self.path_of_all_snapshots):
        #   if isfile(join(self.path_of_all_snapshots, f)):
        #       snapshot_files_name.append(f)

        data_points_with_openface = []
        for datapoint in self.participant.data_points:
            rate = datapoint.rate
            print("Point: " + str(rate.timestamp))
            number_of_snapshots = 0
            start_time_stamp = rate.timestamp - self.period
            end_time_stamp = rate.timestamp - self.margin
            try:
                folder_path = str(self.path_for_saving_datapoint_frames + "\\" + str(self.participant.number) + "\\"
                                  + str(rate.timestamp))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                self.set_webcam_snapshots_for_path(folder_path)

            except OSError:
                print("Creation of the directory %s failed" % folder_path)
                print(OSError.strerror())
            else:
                print("Successfully created the directory %s " % folder_path)
            # Here we copy snapshots in destination directories
            for t in range(len(snapshot_files_timestamp)):
                if start_time_stamp < snapshot_files_timestamp[t] < end_time_stamp:
                    src = str(snapshot_files_name[t])
                    dst = str(folder_path + "\\" + str(t) + ".jpg")
                    rf.extract(src, dst)
                    number_of_snapshots = number_of_snapshots + 1

            open_face_object = Openface(folder_path)
            datapoint.set_openface_object(open_face_object)
            #rate.set_number_of_snapshots(number_of_snapshots)
            data_points_with_openface.append(datapoint)

        participant = self.participant
        participant.set_data_points(data_points_with_openface)
        return participant