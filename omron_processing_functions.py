from datetime import datetime
from pytz import timezone
import time
from omron_object import OmronObject
import os

def read_omron_file(number, path_of_log):
        #print(path_of_log)
        omron_object_output_array = []
        for filename in os.listdir(path_of_log):
            file_dir = path_of_log+'//'+filename
            #print(file_dir)
            file = open(file_dir, "r")
            file_string = file.read()
            #print(file_string)
            file_array_string = file_string.split(
                "-----------------------------------------------------------------------------------------------")
            # print("Length  is : ",len(file_array_string))
            expression_array = []
            neutral_score_array = []
            happiness_score_array = []
            surprise_score_array = []
            anger_score_array = []
            sadness_score_array = []
            timestamp_array = []

            for sample in file_array_string:
                #print(sample)
                expression = ""
                neutral_score = 0
                anger_score = 0
                happiness_score = 0
                surprise_score = 0
                sadness_score = 0

                index = sample.find("Expression:")
                # print(index)
                if (index != -1):
                    try:
                        str = sample[index:len(sample)]
                        splitted_array = str.split(" ")
                        x = splitted_array[len(splitted_array) - 2].split("\n")
                        date_array = x[len(x) - 1].split("/")

                        y = int(date_array[0])
                        mo = int(date_array[1])
                        d = int(date_array[2])

                        time_array = splitted_array[len(splitted_array) - 1].replace("\n", "").split(":")
                        h = int(time_array[0])
                        mi = int(time_array[1])
                        s = int(time_array[2])
                        # print(h,mi,s)

                        # print(splitted_array)

                        tmp = splitted_array[0].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            expression = tmp[1]

                        tmp = splitted_array[2].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            neutral_score = int(tmp[1])

                        tmp = splitted_array[3].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            happiness_score = int(tmp[1])

                        tmp = splitted_array[4].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            surprise_score = int(tmp[1])

                        tmp = splitted_array[5].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            anger_score = int(tmp[1])

                        tmp = splitted_array[6].split(":")
                        if (len(tmp) == 2):
                            tmp[1] = tmp[1].replace(",", "")
                            sadness_score = int(tmp[1])

                        time_stamp = aedt_to_unix_utc(y, mo, d, h, mi, s)
                        expression_array.append(expression)
                        neutral_score_array.append(neutral_score)
                        happiness_score_array.append(happiness_score)
                        surprise_score_array.append(surprise_score)
                        anger_score_array.append(anger_score)
                        sadness_score_array.append(sadness_score)
                        timestamp_array.append(time_stamp)
                        #print(time_stamp)
                    except:
                        # print(sample)
                        print("")
            omron_total_object = OmronObject(participant_number=number, timestamp_array=timestamp_array, expression_array=expression_array, anger_rate_array=anger_score_array,
                                             sadness_rate_array=sadness_score_array, happiness_rate_array=happiness_score_array, surprise_rate_array=surprise_score_array, neutral_rate_array=neutral_score_array, rate=None
                                             )

            return omron_total_object


def aedt_to_unix_utc(y, mo, d, h, mi, s):
        aedt = timezone('Australia/Melbourne')
        utc = timezone('UTC')
        melnourne_date = aedt.localize(datetime(y, mo, d, h, mi, s))
        utc_date = melnourne_date.astimezone(utc)
        fmt = '%Y-%m-%d %H:%M:%S %Z%z'
        #print(melnourne_date.strftime(fmt))
        #print(utc_date.strftime(fmt))
        timestamp = time.mktime(melnourne_date.timetuple())
        return timestamp