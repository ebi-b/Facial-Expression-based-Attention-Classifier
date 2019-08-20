class Rate:
    def __init__(self, timestamp, engagement, engagement_timestamp, challenge, challenge_timestamp):
        self.timestamp = float(timestamp)
        self.engagement = engagement
        self.challenge = challenge
        print(engagement_timestamp)
        try:
            self.engagement_timestamp = float(engagement_timestamp)
            self.challenge_timestamp = float(challenge_timestamp)
        except:
            print("Error in setting engagement timestamp")

    def print_rate(self):
        print("Timestamp:{0}, Eng: {1}, Chal: {2}".format(self.timestamp,self.engagement,self.challenge))

    def set_mini_timestamp(self, timestamp):
        self.mini_timestamp = timestamp