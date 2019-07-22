from openface_object import Openface
from omron_object import OmronObject
import numpy as np
import data_point as dp
import rates
def split_openface_object(openface_object, lenght, margin):
    print("in split_openfacec_object for participant {0} and timestamp {1}".format(openface_object.participant_number ,openface_object.rate.timestamp))
    #openface_object =(Openface) (openface_object)
    if hasattr(openface_object,'snapshot_files_timestamp'):
        if openface_object.snapshot_files_timestamp != []:
            snapshot_files_timestamp = np.array(openface_object.snapshot_files_timestamp)
            max = openface_object.rate.timestamp-margin
            min =snapshot_files_timestamp.min()
            number_of_objects = int((max-min)/lenght)
            mini_openface_objects = []
            print("Len snapshot File Timestamp: ", len(snapshot_files_timestamp))
            print("Len           au_c Array   : ", len(openface_object.au_c_array))
            for j in range(number_of_objects):

                min_timestamp = max - ((j+1) * lenght)
                max_timestamp = max - (j*lenght)
                mini_rate = rates.Rate(openface_object.rate.timestamp, openface_object.rate.engagement,0,openface_object.rate.challenge,0)
                mini_rate.set_mini_timestamp(max_timestamp)
                new_mini_openface_object = Openface(mini_rate, openface_object.path_of_frames,
                                                    openface_object.participant_number, to_mini_points=True)
                for i in range(len(snapshot_files_timestamp)):
                    if min_timestamp < snapshot_files_timestamp[i] <= max_timestamp:
                        new_mini_openface_object.au_c_array.append(openface_object.au_c_array[i])
                        new_mini_openface_object.au_r_array.append(openface_object.au_r_array[i])
                        new_mini_openface_object.gaze_angle_array.append(openface_object.gaze_angle_array[i])
                        new_mini_openface_object.eye_gaze_direction_array.append(openface_object.eye_gaze_direction_array[i])
                        new_mini_openface_object.pose_R_array.append(openface_object.pose_R_array[i])
                        new_mini_openface_object.pose_T_array.append(openface_object.pose_T_array[i])
                        new_mini_openface_object.success_array.append(openface_object.success_array[i])
                        new_mini_openface_object.confidence_array.append(openface_object.confidence_array[i])
                        new_mini_openface_object.face_id_array.append(openface_object.face_id_array[i])
                        new_mini_openface_object.eye_2d_landmarks_array.append(openface_object.eye_2d_landmarks_array[i])
                        new_mini_openface_object.eye_3d_landmarks_array.append(openface_object.eye_3d_landmarks_array[i])
                        new_mini_openface_object.face_2d_landmarks_array.append(openface_object.face_2d_landmarks_array[i])
                        new_mini_openface_object.face_3d_landmarks_array.append(openface_object.face_3d_landmarks_array[i])
                        new_mini_openface_object.p_scale_array.append(openface_object.p_scale_array[i])
                        new_mini_openface_object.rotation_array.append(openface_object.rotation_array[i])
                        new_mini_openface_object.transition_array.append(openface_object.transition_array[i])
                        new_mini_openface_object.non_rigid_shape_parameters_array.append(openface_object.non_rigid_shape_parameters_array[i])
                        new_mini_openface_object.snapshot_files_name.append(openface_object.snapshot_files_name[i])
                        new_mini_openface_object.snapshot_files_timestamp.append(openface_object.snapshot_files_timestamp[i])

                if new_mini_openface_object.snapshot_files_timestamp !=[]:
                    new_mini_timestamps = np.array(new_mini_openface_object.snapshot_files_timestamp)
                    mini_max = new_mini_timestamps.max()
                    mini_min = new_mini_timestamps.min()
                    if mini_max-mini_min > (lenght/3) and len(new_mini_timestamps) > len(snapshot_files_timestamp)/(number_of_objects*3):
                        mini_openface_objects.append(new_mini_openface_object)

            return mini_openface_objects

def split_omron_object(omron_object, lenght, margin):
    print("in split_omron_object for participant {0} and timestamp {1}".format(omron_object.participant_number,
                                                                                   omron_object.rate.timestamp))
    if hasattr(omron_object, 'timestamp_array'):
        if omron_object.timestamp_array != []:
            omron_samples_timestamp = np.array(omron_object.timestamp_array)
            max = omron_object.rate.timestamp - margin
            min = omron_samples_timestamp.min()
            number_of_objects = int((max - min) / lenght)
            mini_omron_objects = []
            print("Len snapshot File Timestamp: ", len(omron_samples_timestamp))
            print("Len           au_c Array   : ", len(omron_object.expression_array))
            for j in range(number_of_objects):
                expression_array = []
                neutral_score_array = []
                happiness_score_array = []
                surprise_score_array = []
                anger_score_array = []
                sadness_score_array = []
                timestamp_array = []
                #new_mini_omron_object = OmronObject(participant_number=omron_object.participant_number, timestamp_array, expression_array, neutral_rate_array,
                 #happiness_rate_array, surprise_rate_array, anger_rate_array, sadness_rate_array, rate = None)
                min_timestamp = max - ((j + 1) * lenght)
                max_timestamp = max - (j * lenght)
                for i in range(len(omron_samples_timestamp)):
                    if min_timestamp < omron_samples_timestamp[i] <= max_timestamp:
                        expression_array.append(omron_object.expression_array[i])
                        neutral_score_array.append(omron_object.neutral_score_array[i])
                        happiness_score_array.append(omron_object.happiness_score_array[i])
                        surprise_score_array.append(omron_object.surprise_score_array[i])
                        anger_score_array.append(omron_object.anger_score_array[i])
                        sadness_score_array.append(omron_object.sadness_score_array[i])
                        timestamp_array.append(omron_object.timestamp_array[i])
                rate_mini = rates.Rate(omron_object.rate.timestamp, omron_object.rate.engagement, 0, omron_object.rate.challenge, 0)
                rate_mini.set_mini_timestamp(max_timestamp)
                #print(rate_mini.timestamp)
                new_mini_omron_object = OmronObject(omron_object.participant_number,
                                                    timestamp_array, expression_array,
                                                    neutral_score_array,
                                                    happiness_score_array,
                                                    surprise_score_array,
                                                    anger_score_array,
                                                    sadness_score_array, rate_mini)

                if new_mini_omron_object.timestamp_array != []:
                    new_mini_timestamps = np.array(new_mini_omron_object.timestamp_array)
                    mini_max = new_mini_timestamps.max()
                    mini_min = new_mini_timestamps.min()
                    if mini_max - mini_min > (lenght / 3) and len(new_mini_timestamps) > len(
                            omron_samples_timestamp) / (number_of_objects * 3):
                        mini_omron_objects.append(new_mini_omron_object)

            return mini_omron_objects

def is_the_same_time_window(mini_rate1, mini_rate2):
    if mini_rate1.mini_timestamp != None and mini_rate2.mini_timestamp !=None:
        t1 = mini_rate1.mini_timestamp
        t2 = mini_rate2.mini_timestamp
        if abs(t1-t2) < 0.01:
            return True
        else:
            return False
    elif mini_rate1.mini_timestamp == None:
        print("mini_rate1 timestamp in is_the_same_window function is None.")
    else:
        print("mini_rate2 timestamp in is_the_same_window function is None.")

def synch_mini_objects(mini_openface_object_array, mini_omron_object_array):
    synched_datapoints = []
    to_remove_mini_omron_obj_array = []
    to_remove_mini_openface_obj_array = []

    for i in range(len(mini_omron_object_array)):
        for j in range(len(mini_openface_object_array)):
            omron_obj = mini_openface_object_array[i]
            openface_obj = mini_openface_object_array[j]
            if is_the_same_time_window(omron_obj.rate, openface_obj.rate):
                new_mini_data_point = dp.DataPoint(openface_obj.participant_number, openface_obj.rate)
                new_mini_data_point.set_openface_object(openface_obj)
                new_mini_data_point.set_omron_object(omron_obj)
                synched_datapoints.append(new_mini_data_point)
                to_remove_mini_omron_obj_array.append(i)
                to_remove_mini_openface_obj_array.append(j)

    to_remove_mini_openface_obj_array.sort(reverse=True)
    to_remove_mini_omron_obj_array.sort(reverse=True)

    for i in to_remove_mini_openface_obj_array:
        del (mini_openface_object_array[i])
    for i in to_remove_mini_omron_obj_array:
        del (mini_omron_object_array[i])

#do rest of code
    return synched_datapoints
