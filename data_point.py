import fanctions_for_splitting_datapoints as fsd

class DataPoint:
    def __init__(self, participant_number, rate):
        self.rate = rate
        self.participant_number = participant_number

    def set_openface_object(self, obj):
        self.openface_object = obj

    def to_mini_data_points(self, lenght):
        mini_data_points =[]
        if self.openface_object != None:
            mini_open_face_objects = fsd.split_openface_object(self.openface_object, lenght)
            if mini_open_face_objects != None and mini_open_face_objects != []:
                for mini_obj in mini_open_face_objects:
                    new_mini_data_point = DataPoint(self.participant_number, self.rate)
                    new_mini_data_point.set_openface_object(mini_obj)
                    mini_data_points.append(mini_obj)

        return mini_data_points