from classifier_facial_expression import FacialExpressionClassifier
from classifier_facial_expression import FacialExpressionClassifier as fec
#classifier = FacialExpressionClassifier("Y:\\Openface_Processed_Frames\\Participant_objects", 5, 5)

participants = fec.load_participants("Y:\\Openface_Processed_Frames\\sample")
for participant in participants:
    #new_data_points = participant.split_to_mini_datapoints(14.5)
    path_of_log = str("omron object\\"+str(participant.number))
    participant.set_omron_objects(path_of_log)
