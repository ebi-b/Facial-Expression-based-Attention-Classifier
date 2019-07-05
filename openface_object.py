import subprocess
import os
import csv
import pickle
from shutil import copyfile

class Openface:

    def __init__(self, rate, path_of_frames):
        print("openface object is created...")
        self.path_of_frames = path_of_frames
        self.rate = rate
        self.dst_dir = "Y:\\Openface Processed Frames\\Folder of CSVc"

    def extract_csv(self):
        path = self.path_of_frames
        print("Path is : " + str(path))

        # print(path)
        cmd = "cd OpenFace_2.0.5_win_x64 && FeatureExtraction.exe -fdir " + path
        subprocess.call(cmd, shell=True)
        src = "OpenFace_2.0.5_;win_x64/processed/" + str(self.rate.timestamp) + ".csv"

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
        if os.path.exists(src):
            dst = str(self.dst_dir) +"\\"+str(self.rate.timestamp) + ".csv"
            copyfile(src, dst)
            self.csv_path = dst

    def openface_csv_read(self):
        with open(self.csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            au_c_array = []
            au_r_array = []
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    au_r = []
                    au_c = []
                    au_r_s = row[679:696]
                    au_c_s = row[696:714]

                    for i in au_r_s:
                        au_r.append(float(i))
                    for i in au_c_s:
                        au_c.append(float(i))
                    au_c_array.append(au_c)
                    au_r_array.append(au_r)
                    line_count += 1
        self.au_r_array=au_r_array
        self.au_c_array=au_c_array
        #return au_r_array, au_c_array
