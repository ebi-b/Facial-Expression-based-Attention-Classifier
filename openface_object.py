import subprocess
import os
import csv
import pickle
from shutil import copyfile
from os import listdir
from os.path import isfile, join
from termcolor import colored

class Openface:

    def __init__(self, rate, path_of_frames, participant_number):
        print("openface object is created...")
        self.participant_number = participant_number
        self.path_of_frames = path_of_frames
        self.rate = rate
        self.snapshot_files_name = []
        self.snapshot_files_timestamp = []
        self.set_frame_name_and_timestamp()
        self.dst_dir = "Y:\\Openface_Processed_Frames\\Folder_of_CSVs"+"\\"+str(participant_number)
        self.csv_path = ""
        self.extract_csv()
        if self.csv_path == "":
            tmpstr = "Folder {0} for Participant {1} is Empty Check it...".format(rate.timestamp,participant_number)
            print(colored(tmpstr, 'red'))
        else:
            self.openface_csv_read()


    def extract_csv(self):
        print("Extracting Openface CSVs...")
        path = self.path_of_frames
        #print("Path is : " + str(path))
        dst = str(self.dst_dir) + "\\" + str(self.rate.timestamp) + ".csv"
        if os.path.exists(dst):
            tmpstr="CSV {0} for Participant {1} is already exists".format(self.rate.timestamp,self.participant_number)
            self.csv_path = dst
            print(colored(tmpstr, 'green'))
        # print(path)
        else:
            cmd = "cd OpenFace_2.0.5_win_x64 && FeatureExtraction.exe -fdir " + path
            subprocess.call(cmd, shell=True)
            src = "OpenFace_2.0.5_win_x64\\processed\\" + str(self.rate.timestamp) + ".csv"

            if not os.path.exists(self.dst_dir):
                os.makedirs(self.dst_dir)
            if os.path.exists(src):
                dst = str(self.dst_dir) +"\\"+str(self.rate.timestamp) + ".csv"
                copyfile(src, dst)
                self.csv_path = dst

    def openface_csv_read(self):
        print("Reading Openface CSVs...")
        with open(self.csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            au_c_array = []
            au_r_array = []
            gaze_angle_array = []
            eye_gaze_direction_array = []
            pose_T_array = []
            pose_R_array = []
            success_array = []
            confidence_array = []
            face_id_array = []
            eye_2d_landmarks_array = []
            eye_3d_landmarks_array = []

            face_2d_landmarks_array = []
            face_3d_landmarks_array = []

            p_scale_array = []
            rotation_array = []
            transition_array = []
            non_rigid_shape_parameters_array = []

            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    au_r = []
                    au_c = []

                    eye_gaze_direction = []
                    gaze_angle = []

                    pose_T = []
                    pose_R = []

                    success = []
                    confidence = []
                    face_id = []

                    eye_2d_landmarks = []
                    eye_3d_landmarks = []

                    face_2d_landmarks = []
                    face_3d_landmarks = []

                    p_scale = []
                    rotation = []
                    transition = []
                    non_rigid_shape_parameters = []

                    au_r_s = row[679:696]
                    au_c_s = row[696:714]

                    success_s = int(row[4])
                    confidence_s = row[3]
                    face_id_s = row[1]

                    eye_2d_landmarks_s = row[13:125]
                    eye_3d_landmarks_s = row[125:293]
                    # print(eye_3d_landmarks_s, '\n')

                    face_2d_landmarks_s = row[299:434]
                    face_3d_landmarks_s = row[435:639]

                    p_scale_s = row[639]
                    rotation_s = row[640:643]

                    transition_s = row[643:645]

                    non_rigid_shape_parameters_s = row[645:679]
                    #print(non_rigid_shape_parameters_s, '\n')

                    eye_0_gaze_direction_x_t = row[5]
                    eye_0_gaze_direction_y_t = row[6]
                    eye_0_gaze_direction_z_t = row[7]

                    eye_1_gaze_direction_x_t = row[8]
                    eye_1_gaze_direction_y_t = row[9]
                    eye_1_gaze_direction_z_t = row[10]

                    eye_gaze_direction_s = [eye_0_gaze_direction_x_t, eye_0_gaze_direction_y_t,
                                            eye_0_gaze_direction_z_t,
                                            eye_1_gaze_direction_x_t, eye_1_gaze_direction_y_t,
                                            eye_1_gaze_direction_z_t]

                    gaze_angle_x_t = row[11]
                    gaze_angle_y_t = row[12]
                    gaze_angle_s = [gaze_angle_x_t, gaze_angle_y_t]

                    pose_Tx_t = row[293]
                    pose_Ty_t = row[294]
                    pose_Tz_t = row[295]
                    pose_T_s = [pose_Tx_t, pose_Ty_t, pose_Tz_t]

                    pose_Rx_t = row[296]
                    pose_Ry_t = row[297]
                    pose_Rz_t = row[298]
                    pose_R_s = [pose_Rx_t, pose_Ry_t, pose_Rz_t]

                    for i in au_r_s:
                        au_r.append(float(i))

                    for i in au_c_s:
                        au_c.append(float(i))

                    for i in eye_gaze_direction_s:
                        eye_gaze_direction.append(float(i))

                    for i in gaze_angle_s:
                        gaze_angle.append(float(i))

                    for i in pose_R_s:
                        pose_R.append(float(i))

                    for i in pose_T_s:
                        pose_T.append(float(i))

                    success.append(int(success_s))
                    confidence.append(float(confidence_s))
                    face_id.append(int(face_id_s))

                    for i in eye_2d_landmarks_s:
                        eye_2d_landmarks.append(float(i))

                    for i in eye_3d_landmarks_s:
                        eye_3d_landmarks.append(float(i))

                    for i in face_2d_landmarks_s:
                        face_2d_landmarks.append(float(i))

                    for i in face_3d_landmarks_s:
                        face_3d_landmarks.append(float(i))

                    p_scale.append(float(p_scale_s))

                    for i in rotation_s:
                        rotation.append(float(i))

                    for i in transition_s:
                        transition.append(float(i))

                    for i in non_rigid_shape_parameters_s:
                        non_rigid_shape_parameters.append(float(i))

                    au_c_array.append(au_c)
                    au_r_array.append(au_r)
                    gaze_angle_array.append(gaze_angle)
                    eye_gaze_direction_array.append(eye_gaze_direction)
                    pose_R_array.append(pose_R)
                    pose_T_array.append(pose_T)
                    success_array.append(success)
                    confidence_array.append(confidence)
                    face_id_array.append(face_id)
                    eye_2d_landmarks_array.append(eye_2d_landmarks)
                    eye_3d_landmarks_array.append(eye_3d_landmarks)
                    face_2d_landmarks_array.append(face_2d_landmarks)
                    face_3d_landmarks_array.append(face_3d_landmarks)
                    p_scale_array.append(p_scale)
                    rotation_array.append(rotation)
                    transition_array.append(transition)
                    non_rigid_shape_parameters_array.append(non_rigid_shape_parameters)

                    line_count += 1

        # success = []
        # confidence = []
        # face_id = []
        # eye_2d_landmarks = []
        # eye_3d_landmarks = []
        # face_2d_landmarks = []
        # face_3d_landmarks = []
        # p_scale = []
        # rotation = []
        # transition = []
        # non_rigid_shape_parameters = []

        # print(len(eye_2d_landmarks_array))

    def set_frame_name_and_timestamp(self):
        for f in listdir(self.path_of_frames):
            if isfile(join(self.path_of_frames, f)):
                self.snapshot_files_name.append(f)

        for name in self.snapshot_files_name:
            tmp = name.replace(".jpg", "")
            self.snapshot_files_timestamp.append(float(tmp))
            #print("File: " + tmp)

    def to_string(self):
        return "Participant No : {0}| Timestamp : {1}| len gaze angle array: {3} ".format(self.participant_number, self.rate.timestamp, len(self.gaze_angle_array))
