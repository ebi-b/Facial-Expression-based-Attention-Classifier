from openface_object import Openface
import numpy as np
def split_openface_object(openface_object, lenght):
    print("in split_openfacec_object for participant {0} and timestamp {1}".format(openface_object.participant_number ,openface_object.rate.timestamp))
    #openface_object =(Openface) (openface_object)
    if hasattr(openface_object,'snapshot_files_timestamp'):
        if openface_object.snapshot_files_timestamp != []:
            snapshot_files_timestamp = np.array(openface_object.snapshot_files_timestamp)
            max = snapshot_files_timestamp.max()
            min =snapshot_files_timestamp.min()
            number_of_objects = int((max-min)/lenght)
            mini_openface_objects = []
            print("Len snapshot File Timestamp: ", len(snapshot_files_timestamp))
            print("Len           au_c Array   : ", len(openface_object.au_c_array))
            for j in range(number_of_objects):
                new_mini_openface_object = Openface(openface_object.rate, openface_object.path_of_frames, openface_object.participant_number, to_mini_points = True)
                min_timestamp = max - ((j+1) * lenght)
                max_timestamp = max - (j*lenght)
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
