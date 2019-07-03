import array
import csv
import os
from decimal import Decimal
from rates import Rate
from interruption import Interruptions

class Participant:

    def __init__(self, number, date, age, gender, path_of_logs):
        self.number = number
        self.date = date
        self.age = age
        self.gender = gender
        self.rates = []
        self.interruptions = []
        self.set_rates(path_of_logs)

    #This function set user engagement and challenge inputs in Participant Object
    def set_rates(self, path):
        lastInteruptionTimeStamp=""
        lastChallengeTimeStamp=""
        lastEngagementTimeStamp=""
        lastEngagementValue=0
        lastChallengeValue=0
        lastSubmitTimeStamp=""
        lastInteruptionTimeStamp=""
        print("Setting Rates for Participant {0}.".format(self.number))
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                #tmp1 = "Sliders CSVs/Engagement and Challenge/" + filename + "E.csv"
                #Pallete_pp_CSV = open(tmp1, "a")
                #tmp2 = "Sliders CSVs/Interuptions/" + filename + "I.csv"
                #Interuption_pp_CSV = open(tmp2, "a")
                #sliders_pp_writer = csv.writer(Pallete_pp_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #interruption_pp_writer = csv.writer(Interuption_pp_CSV, delimiter=',')
                sliderLogFile = open(path + '/' + filename, 'r')
                lenght_of_file = len(open(path + '/' + filename).readlines())

                lastChallengeTimeStamp = '0'
                lastChallengeValue = 0
                lastSubmitTimeStamp = '0'
                lastInteruptionTimeStamp = '0'
                for i in range(lenght_of_file):
                    line = sliderLogFile.readline()
                    # print(str(i) +' : ' +line)
                    if line.find("Challenge") != -1:
                        try:
                            lastChallengeStringValue = Participant.find_value(line)
                            lastChallengeValue = Participant.map_values(lastChallengeStringValue, 'c')
                            lastChallengeTimeStamp = Participant.find_time_stamp(line)
                        except:
                            print("Exception: " + line)
                    if line.find("Engagement") != -1:
                        try:
                            lastEngagementStringValue = Participant.find_value(line)
                            lastEngagementValue = Participant.map_values(lastEngagementStringValue, 'e')
                            lastEngagementTimeStamp = Participant.find_time_stamp(line)
                        except:
                            print("Exception: " + line)

                    if line.find("Submit") != -1:
                        #print(line)
                        lastSubmitTimeStamp = Participant.find_time_stamp(line)
                        tmp_rate = Rate(lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp,
                                        lastChallengeValue, lastChallengeTimeStamp)
                        self.rates.append(tmp_rate)

                        #sliders_writer.writerow(
                        #   [lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp, lastChallengeValue,
                        #    lastChallengeTimeStamp])
                        #sliders_pp_writer.writerow(
                        #   [lastSubmitTimeStamp, lastEngagementValue, lastEngagementTimeStamp, lastChallengeValue,
                        #    lastChallengeTimeStamp])
                        # writeString=lastSubmitTimeStamp+','+str(lastEngagementValue)+','+lastEngagementTimeStamp+','+str(lastChallengeValue)+','+lastChallengeTimeStamp+';'
                        # PalleteCSV.write(writeString)

                    if line.find("Interuption") != -1:
                        lastInteruptionTimeStamp = Participant.find_time_stamp(line)
                        tmp_interruption= Interruptions(lastInteruptionTimeStamp,'--Reason--')
                        #interruption_writer.writerow([lastInteruptionTimeStamp, '--Reason-'])
                        #interruption_pp_writer.writerow([lastInteruptionTimeStamp, '--Reason-'])
                        # writeString=lastInteruptionTimeStamp+',--;'
                        # InteruptionCSV.write(writeString)

                continue
            else:
                continue

    #This function find values of engagement or challenge in a line of pallete log.
    @staticmethod
    def find_value(s):
        j1 = s.split('{')
        j2 = j1[1].split('[')
        j3 = j2[1].split(']')
        value = j3[0]
        return value

    # This function find timestamp in a line of pallete log.
    @staticmethod
    def find_time_stamp(s):
        j = s.split("{")
        d = j[0].split(" ")
        time = d[2]
        decimal_time = Decimal(time)
        decimal_time = decimal_time / 1000
        return str(decimal_time)

    #This function map the values of logs to real values.
    @staticmethod
    def map_values(sv, t):
        v = float(sv)
        minIndex = 0
        cLevel = array.array('f',
                             [0, 0.39, 5.1, 10.98, 17.25, 21.18, 26.27, 32.55, 38.04, 43.53, 48.24, 53.73, 60, 64.71,
                              69.41, 75.69, 82.35, 87.06, 92.55, 99.22, 99.61])
        eLevel = array.array('f', [0, 0.39, 3.14, 9.02, 15.29, 20, 25.1, 30.2, 36.47, 41.57, 46.67, 52.55, 57.65, 63.53,
                                   69.02, 74.12, 80.78, 85.1, 91.37, 99.22, 99.61])
        if t == 'c':
            mindif = 1000
            minIndex = 0
            for i in range(0, 21):
                dif = abs(cLevel[i] - v)
                if dif < mindif:
                    minIndex = i
                    mindif = dif

        if t == 'e':
            mindif = 1000
            minIndex = 0
            for i in range(0, 21):
                # print(i)
                dif = abs(eLevel[i] - v)
                if dif < mindif:
                    minIndex = i
                    mindif = dif

        realValue = minIndex * 5
        return realValue