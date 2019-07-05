from participant import Participant
import os
import csv
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import pickle

class Openface:

    def __init__(self, participant, period, margin):
        self.participant = Participant(participant)
        self.period = period
        self.path_of_snapshots = participant.path_of_snapshots
        self.margin = margin
        self.path_for_saving_snapshots = ""
        self.path_for_saving_datapoint_frames="Y:\\Openface Processed Frames\\Folders of Datapoints"

    def preparation_for_analysis(self):
        snapshot_files_name = []
        for f in listdir(self.path_of_snapshots):
            if isfile(join(self.path_of_snapshots, f)):
                snapshot_files_name.append(f)

        new_data_points = []
        for rate in self.participant.rates:
            print("Point: " + str(rate.timestamp))
            number_of_snapshots = 0
            start_time_stamp = rate.timestamp - self.period
            end_time_stamp = rate.timestamp
            try:
                folder_path = str(self.path_for_saving_datapoint_frames + "\\" + str(self.participant.number) + "\\" + str(rate.timestamp))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                self.set_webcam_snapshots_for_path(folder_path)

            except OSError:
                print("Creation of the directory %s failed" % folder_path)
                print(OSError.strerror())
            else:
                print("Successfully created the directory %s " % folder_path)
            # Here we copy snapshots in destination directories
            for t in snapshotFilesTimeStamp:
                if start_time_stamp < t < end_time_stamp:
                    src = str(path_of_snapshots + "\\" + str(t) + ".jpg")
                    dst = str(folder_path + "\\" + str(t) + ".jpg")
                    copyfile(src, dst)
                    number_of_snapshots = number_of_snapshots + 1
            rate.set_number_of_snapshots(number_of_snapshots)
            new_data_points.append(rate)