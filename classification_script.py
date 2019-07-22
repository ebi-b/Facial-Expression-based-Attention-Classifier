from classifier_facial_expression import FacialExpressionClassifier
import pickle
from classifier_facial_expression import FacialExpressionClassifier as fec
from termcolor import colored

#classifier = FacialExpressionClassifier("Y:\\Openface_Processed_Frames\\Participant_objects", 5, 5)

participants = fec.load_participants("Y:\\Openface_Processed_Frames\\sample")
for participant in participants:
    path_of_log = str("omron object\\" + str(participant.number))
    participant.set_omron_objects(path_of_log)
    for lenght in range(15, 61, 5):
        try:
            participant.split_to_mini_datapoints(lenght)
            filehandler = open("Y:\\Openface_Processed_Frames\\mini_participants\\"+str(lenght) +"\\"+ str(participant.number) + ".obj",
                               'wb')
            pickle.dump(participant, filehandler)

        except:
            str="Error in participant {0} and lenght {1}".format(participant.number, lenght)
            print(colored(str , 'red'))


