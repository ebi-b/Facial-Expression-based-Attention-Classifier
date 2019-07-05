class DataPoint:
    def __init__(self, participant_number, rate):
        self.rate = rate
        self.participant_number = participant_number

    def set_openface_object(self, obj):
        self.openface_object = obj

