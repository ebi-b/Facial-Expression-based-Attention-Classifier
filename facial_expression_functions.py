import  numpy as np

@staticmethod
def calculate_gaze_angle_parameters(point):
    print("In Calculating gaze Aneles")
    avg_movement_gaze_angle, gaze_angle_avg,gaze_angle_std = [], [], []
    if hasattr(point.openface_object, 'gaze_angle_array'):
        gaze_angle_array = point.openface_object.gaze_angle_array
        print(gaze_angle_array)
        eye_gaze_direction_array = point.openface_object.eye_gaze_direction_array
        number = 0
        dst = np.array([0, 0])
        for i in range(1, len(gaze_angle_array)):
            if sum(gaze_angle_array[i]) != 0 and sum(gaze_angle_array[i - 1]) != 0:
                dst += abs(gaze_angle_array[i] - gaze_angle_array[i - 1])
                number += 1
        avg_movement_gaze_angle = dst / number

        new_gaze_angle_array = []
        for i in range(len(gaze_angle_array)):
            if sum(gaze_angle_array[i]) != 0:
                new_gaze_angle_array.append(gaze_angle_array[i])

        new_gaze_angle_array = np.array(new_gaze_angle_array)
        gaze_angle_avg = new_gaze_angle_array.mean(axis=0)
        gaze_angle_std = new_gaze_angle_array.std(axis=0)

        print(avg_movement_gaze_angle, gaze_angle_avg, gaze_angle_std)
    return avg_movement_gaze_angle, gaze_angle_avg, gaze_angle_std

@staticmethod
def calculate_action_units_parameters(point, period):
        print("Calculating Metrics in processed_data_points for participant {0} and datapoint {1}...".format(point.participant_number, point.rate.timestamp))
        new_array_c = []
        new_array_r = []
        au_c_avg, au_c_std, au_r_avg, au_r_std = [], [], [], []
        if hasattr(point.openface_object, 'au_c_array'):
            y=0
            for row in point.openface_object.au_c_array:
                if float(point.openface_object.rate.timestamp)-float(point.openface_object.snapshot_files_timestamp[y]) < period:
                    if (sum(row) != 0):
                        #print("Row is: ",len(row))
                        #print("New array is: ",len(new_array_c))
                        #np.append(new_array_c,[row], axis=0)
                        new_array_c.append(row)
                y+=1
                #print(new_array_c.s)
            new_array_c = np.array(new_array_c)
            if len(new_array_c) == 0:
                au_c_avg = np.zeros(18)
                au_c_std = np.zeros(18)
                #print("au_c for point {0} of participant {1} is {3}: ".format(point.rate.timestamp,point.participant_number,new_array_c))
            else:
                au_c_avg = new_array_c.mean(axis=0)
                au_c_std = new_array_c.std(axis=0)

            y=0
            for row in point.openface_object.au_r_array:
                if float(point.openface_object.rate.timestamp) - float(
                        point.openface_object.snapshot_files_timestamp[y]) < period:
                    if (sum(row) != 0):
                        #np.append(new_array_r,[row], axis=0)
                        new_array_r.append(row)
                y+=1
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
            #gaze_angle_movements_avg, gaze_angle_avg, gaze_angle_std = self.calculate_gaze_angle_parameters(openFaceObject= point.openface_object)
            return au_c_avg, au_c_std, au_r_avg, au_r_std
                #, gaze_angle_movements_avg, gaze_angle_avg, gaze_angle_std

        else:
            return [], [], [], []
                #, [], [], []

def calcualate_facial_expression_parameters(point, period):
    #au_c_avg, au_c_std, au_r_avg, au_r_std = calculate_action_units_parameters(point,period)
    gaze_avg_movements, gaze_avg, gaze_std = calculate_gaze_angle_parameters(point.openface_object)