import os
import pickle
from classifier_facial_expression import FacialExpressionClassifier
from classifier_for_challenge import FacialExpressionClassifierChal
from classifier_for_engagement import FacialExpressionClassifierEng
path = "C:\\mini_participants\\120"
participants = []
for filename in os.listdir(path):
        file_pi2 = open(str(path) + "\\" + str(filename), 'rb')
        participant = pickle.load(file_pi2)
        print("Participant {0} is Loaded.".format(filename))
        if participant.number != 55:
            participants.append(participant)
        # filehandler = open("participants.obj", 'wb')
        # pickle.dump(participants, filehandler)
        # return participants

classifier = FacialExpressionClassifier(participants, 5, 5)