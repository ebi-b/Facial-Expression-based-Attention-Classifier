class Rate:
    def __init__(self, timestamp, engagement, engagement_timestamp, challenge, challenge_timestamp):
        self.timestamp = timestamp
        self.engagement = engagement
        self.challenge = challenge
        self.engagement_timestamp = engagement_timestamp
        self.challenge_timestamp = challenge_timestamp

    def print_rate(self):
        print("Timestamp:{0}, Eng: {1}, Chal: {2}".format(self.timestamp,self.engagement,self.challenge))
