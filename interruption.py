class Interruptions:
    def __init__(self, timestamp, reason):
        self.timestamp = timestamp
        self.reason = reason

    def set_reason(self, reason):
        self.reason = reason
