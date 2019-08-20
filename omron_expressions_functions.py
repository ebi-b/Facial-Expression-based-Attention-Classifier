import numpy as np
import statistics
def calculate_omron_expression_array_metric(point):
    exp_numbers = np.zeros(5)
    avg_exp = np.zeros(5)
    if hasattr(point.omron_object, 'expression_array'):
        expression_array = point.omron_object.expression_array
        for i in range(len(expression_array)):
            if expression_array[i] == 'Neutral':
                exp_numbers[0] += 1
            if expression_array[i] == 'Happiness':
                exp_numbers[1] += 1
            if expression_array[i] == 'Surprise':
                exp_numbers[2] += 1
            if expression_array[i] == 'Anger':
                exp_numbers[3] += 1
            if expression_array[i] == 'Sadness':
                exp_numbers[4] += 1
        avg_exp = np.zeros(5)
        if sum(exp_numbers) != 0:
            avg_exp = exp_numbers/sum(exp_numbers)
        max_a = np.zeros(5)
        max_v = 0
        for i in range(len(exp_numbers)):
                if exp_numbers[i] > max_v:
                    max_a = np.zeros(5)
                    max_a[i] = 1
                    max_v = exp_numbers[i]
                elif exp_numbers[i] == max_v:
                    max_a[i] = 1
        return avg_exp, max_a

def calculate_omron_emotions_score_array_metric(point):

    avg_sadness_rate, std_sadness_rate, median_sadness_rate, avg_anger_rate, std_anger_rate, median_anger_rate \
        , avg_surprise_rate, std_surprise_rate, median_surprise_rate, avg_happines_rate, std_happiness_rate, \
    median_happiness_rate, avg_neutral_rate, std_neutral_rate, median_neutral_rate= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    if hasattr(point.omron_object, 'neutral_score_array'):
        neutral_rate_array = point.omron_object.neutral_score_array
        if(len(neutral_rate_array )!=0):
            avg_neutral_rate = sum(neutral_rate_array)/len(neutral_rate_array)
            std_neutral_rate = statistics.stdev(neutral_rate_array)
            median_neutral_rate = statistics.median(neutral_rate_array)

    if hasattr(point.omron_object, 'happiness_score_array'):
        happiness_rate_array = point.omron_object.happiness_score_array
        if (len(happiness_rate_array) != 0):
            avg_happines_rate = sum(happiness_rate_array) / len(happiness_rate_array)
            std_happiness_rate = statistics.stdev(happiness_rate_array)
            median_happiness_rate = statistics.median(happiness_rate_array)

    if hasattr(point.omron_object, 'surprise_score_array'):
        surprise_rate_array = point.omron_object.surprise_score_array
        if (len(surprise_rate_array) != 0):
            avg_surprise_rate = sum(surprise_rate_array) / len(surprise_rate_array)
            std_surprise_rate = statistics.stdev(surprise_rate_array)
            median_surprise_rate = statistics.median(surprise_rate_array)

    if hasattr(point.omron_object, 'anger_score_array'):
        anger_rate_array = point.omron_object.anger_score_array
        if (len(anger_rate_array) != 0):
            avg_anger_rate = sum(anger_rate_array) / len(anger_rate_array)
            std_anger_rate = statistics.stdev(anger_rate_array)
            median_anger_rate = statistics.median(anger_rate_array)

    if hasattr(point.omron_object, 'sadness_score_array'):
        sadness_rate_array = point.omron_object.sadness_score_array
        if (len(sadness_rate_array) != 0):
            avg_sadness_rate = sum(sadness_rate_array) / len(sadness_rate_array)
            std_sadness_rate = statistics.stdev(sadness_rate_array)
            median_sadness_rate = statistics.median(sadness_rate_array)

    return avg_sadness_rate, std_sadness_rate, median_sadness_rate,avg_anger_rate, std_anger_rate, median_anger_rate\
        , avg_surprise_rate,std_surprise_rate, median_surprise_rate, avg_happines_rate, std_happiness_rate, \
        median_happiness_rate, avg_neutral_rate,std_neutral_rate, median_neutral_rate
