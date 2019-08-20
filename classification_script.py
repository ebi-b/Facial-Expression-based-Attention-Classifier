from classifier_facial_expression import FacialExpressionClassifier
import pickle
from classifier_facial_expression import FacialExpressionClassifier as fec
from termcolor import colored
import sys

def remove_irrelevant_data_points(participants):
    for participant in participants:
        if participant.number == 7:
            del(participant.data_points[10])
            del (participant.data_points[6])
            del (participant.rates[10])
            del (participant.rates[6])

        if participant.number == 12:
            del (participant.data_points[3])
            del (participant.data_points[2])
            del (participant.rates[3])
            del (participant.rates[2])

        if participant.number == 15:
            del (participant.data_points[0])
            del (participant.rates[0])

        if participant.number == 20:
            del (participant.data_points[4])
            del (participant.rates[4])
    return participants

def splitting_participants(margin, step):
    participants = fec.load_participants("C:\\Participant_objects")
    participants = remove_irrelevant_data_points(participants)
    for participant in participants:
        margin = 15
        step = 5
        path_of_log = str("omron object\\" + str(participant.number))
        participant.set_omron_objects(path_of_log, margin)
        for lenght in range(60, 181, 60):
            step = int(lenght/2)
            for marginx in range(15, lenght+margin, step):
                try:
                    participant.split_to_mini_datapoints(lenght, marginx)
                    filehandler = open("C:\\mini_participants\\"+str(lenght)+"\\"+ str(participant.number)+"-"+str(marginx) + ".obj",
                                       'wb')
                    pickle.dump(participant, filehandler)

                except:
                    estr="Error in participant {0} and lenght {1}".format(participant.number, lenght)
                    print(colored(estr , 'red'))
#args = sys.argv[1]
#participants = args[1]
#classifier = FacialExpressionClassifier(participants, 5, 5)
splitting_participants(15,5)