import functions_for_splitting_datapoints as fsd
from omron_object import OmronObject


class DataPoint:
    def __init__(self, participant_number, rate):
        self.rate = rate
        self.participant_number = participant_number

    def set_openface_object(self, obj):
        self.openface_object = obj

    def to_mini_data_points(self, lenght, margin):
        mini_data_points =[]
        mini_open_face_objects =[]
        mini_omron_objects=[]

        if self.omron_object != None:
            mini_omron_objects = fsd.split_omron_object(self.omron_object, lenght, margin)

        if self.openface_object != None:
            mini_open_face_objects = fsd.split_openface_object(self.openface_object, lenght, margin)



        if mini_open_face_objects != None and mini_open_face_objects != []:
            if mini_omron_objects != None and mini_omron_objects != []:
                mini_data_points = fsd.synch_mini_objects(mini_open_face_objects, mini_omron_objects)

        return mini_data_points

    def generate_omron_objects(self, total_omron_object, period, margin):
        if total_omron_object.timestamp_array != []:
            expression_array = []
            neutral_score_array = []
            happiness_score_array = []
            surprise_score_array = []
            anger_score_array = []
            sadness_score_array = []
            timestamp_array = []
            for i in range(len(total_omron_object.timestamp_array)):
                timestamp = total_omron_object.timestamp_array[i]
                if self.rate.timestamp - period-margin <= timestamp <= self.rate.timestamp-margin:
                    expression_array.append(total_omron_object.expression_array[i])
                    neutral_score_array.append(total_omron_object.neutral_score_array[i])
                    happiness_score_array.append(total_omron_object.happiness_score_array[i])
                    surprise_score_array.append(total_omron_object.surprise_score_array[i])
                    anger_score_array.append(total_omron_object.anger_score_array[i])
                    sadness_score_array.append(total_omron_object.sadness_score_array[i])
                    timestamp_array.append(timestamp)

            self.omron_object = OmronObject(rate=self.rate, participant_number=self.participant_number, expression_array=expression_array, neutral_rate_array=neutral_score_array, happiness_rate_array=happiness_score_array,surprise_rate_array=surprise_score_array,anger_rate_array=anger_score_array,sadness_rate_array=sadness_score_array,timestamp_array=timestamp_array)

    def set_omron_object(self, omron_object):
        self.omron_object = omron_object
