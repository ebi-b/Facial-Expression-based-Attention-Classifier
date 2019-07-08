import subprocess
import os
import csv
import pickle
from shutil import copyfile

class Openface:

    def __init__(self, rate, path_of_frames, participant_number):
        print("openface object is created...")
        self.path_of_frames = path_of_frames
        self.rate = rate
        self.dst_dir = "Y:\\Openface_Processed_Frames\\Folder_of_CSVc"+str(participant_number)
        self.extract_csv()
        self.openface_csv_read()
        self.participant_number = participant_number

    def extract_csv(self):
        print("Extracting Openface CSVs...")
        path = self.path_of_frames
        #print("Path is : " + str(path))

        # print(path)
        cmd = "cd OpenFace_2.0.5_win_x64 && FeatureExtraction.exe -fdir " + path
        subprocess.call(cmd, shell=True)
        src = "OpenFace_2.0.5_;win_x64\\processed\\" + str(self.rate.timestamp) + ".csv"

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

            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    au_r = []
                    au_c = []
                    eye_gaze_direction = []
                    gaze_angle = []
                    pose_T = []
                    pose_R = []

                    au_r_s = row[679:696]
                    au_c_s = row[696:714]

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

                    au_c_array.append(au_c)
                    au_r_array.append(au_r)
                    gaze_angle_array.append(gaze_angle)
                    eye_gaze_direction_array.append(eye_gaze_direction)
                    pose_R_array.append(pose_R)
                    pose_T_array.append(pose_T)

                    line_count += 1
        # au_r_array = au_r_array
        # au_c_array = au_c_array
        self.au_c_array = au_c_array
        self.au_r_array = au_r_array
        self.pose_R_array = pose_R_array
        self.pose_T_array = pose_T_array
        self.eye_gaze_direction_array = eye_gaze_direction_array
        self.gaze_angle_array = gaze_angle_array
        print(pose_R_array)

