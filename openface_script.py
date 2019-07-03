import subprocess
import os

path="Y:\P9\webcamSnapshot"
cmd = "cd.. && cd OpenFace_2.0.5_win_x64 && FeatureExtraction.exe -fdir "+path
subprocess.call(cmd, shell=True)