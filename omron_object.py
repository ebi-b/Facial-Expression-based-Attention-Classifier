from datetime import datetime
from pytz import timezone
import time

class OmronObject:
    def __init__(self, participant_number, timestamp_array, expression_array, neutral_rate_array,
                 happiness_rate_array, surprise_rate_array, anger_rate_array, sadness_rate_array):
        self.participant_number = participant_number
        self.expression_array = expression_array
        self.neutral_score_array = neutral_rate_array
        self.happiness_score_array = happiness_rate_array
        self.surprise_score_array = surprise_rate_array
        self.anger_score_array = anger_rate_array
        self.sadness_score_array = sadness_rate_array
        self.timestamp_array = timestamp_array
