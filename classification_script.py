from classifier_facial_expression import FacialExpressionClassifier
from classifier_facial_expression import FacialExpressionClassifier as fec
#classifier = FacialExpressionClassifier("Y:\\Openface_Processed_Frames\\Participant_objects", 5, 5)

participants = fec.load_participants("Y:\\Openface_Processed_Frames\\Participant_objects")
for participant in participants:
    participant.split_to_mini_datapoints(14.5)