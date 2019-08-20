import facial_expression_functions as fef
import omron_expressions_functions as oef

def prepare_features(data_points):
    chal_rates, eng_rates,eng_labels, chal_labels, quadrant_labels, au_cr, p_numbers = [], [], [], [], [], [], []
    # if type(data_points)!= 'list':
    #    data_points = [data_points]
    for i in range(len(data_points)):
        if type(data_points[i]) == 'list':
            for j in range(len(data_points[i])):
                point = data_points[i][j]
        else:
            point = data_points[i]

        # if hasattr(data_points[i], 'rate'):
        # for point in data_points[i]:
        quadrant_label, eng_label, chal_label = calculate_labels(point)
        eng_rate = point.rate.engagement
        chal_rate = point.rate.challenge
        try:
                    tmp = []
                    au_c_avg, au_c_std, au_r_avg, au_r_std = fef.calculate_action_units_parameters(point, 30)
                    gaze_avg_movements, gaze_avg, gaze_std = fef.calculate_gaze_angle_parameters(point)
                    avg_movement_head_pose, head_pose_avg, head_pose_std = fef.calculate_head_pose(point)
                    avg_movement_pitch_roll_yaw, pitch_roll_yaw_avg, pitch_roll_yaw_std = fef.calculate_pitch_roll_yaw(
                        point)
                    avg_sadness_rate, std_sadness_rate, median_sadness_rate, avg_anger_rate, std_anger_rate, median_anger_rate \
                        , avg_surprise_rate, std_surprise_rate, median_surprise_rate, avg_happines_rate, std_happiness_rate, \
                    median_happiness_rate, avg_neutral_rate, std_neutral_rate, median_neutral_rate = \
                        oef.calculate_omron_emotions_score_array_metric(point)

                    avg_exp, max_a = oef.calculate_omron_expression_array_metric(point)

                    p_scale_avg, p_scale_std, rotation_avg, rotation_std, transition_avg, transition_std, \
                    non_rigid_shape_parameters_avg, non_rigid_shape_parameters_std = fef.calcualate_shape_parameters(
                        point)

                    #print("MAMOOD MAMAMD")

                    if sum(au_c_avg) == 0:
                        abas = 1
                    if True:
                        eng_labels.append(eng_label)
                        chal_labels.append(chal_label)
                        eng_rates.append(eng_rate)
                        chal_rates.append(chal_rate)

                        quadrant_labels.append(quadrant_label)
                        p_numbers.append(point.participant_number)

                        rates = [eng_rate, chal_rate]
                        tmp.extend(rates)
                        tmp.extend(au_r_avg)
                        tmp.extend(au_r_std)
                        tmp.extend(au_c_avg)
                        tmp.extend(au_c_std)
                        tmp.extend(gaze_avg_movements)
                        tmp.extend(gaze_avg)
                        tmp.extend(gaze_std)

                        tmp.extend(avg_movement_head_pose)
                        tmp.extend(head_pose_avg)
                        tmp.extend(head_pose_std)

                        tmp.extend(avg_movement_pitch_roll_yaw)
                        tmp.extend(pitch_roll_yaw_avg)
                        tmp.extend(pitch_roll_yaw_std)

                        # OMRON FEATURES::

                        omron_emotion_array = [avg_sadness_rate, std_sadness_rate, median_sadness_rate, avg_anger_rate,
                                               std_anger_rate, median_anger_rate \
                            , avg_surprise_rate, std_surprise_rate, median_surprise_rate, avg_happines_rate,
                                               std_happiness_rate, \
                                               median_happiness_rate, avg_neutral_rate, std_neutral_rate,
                                               median_neutral_rate]

                        tmp.extend(omron_emotion_array)
                        tmp.extend(avg_exp)
                        tmp.extend(max_a)
                        tmp.extend(p_scale_avg)
                        tmp.extend(p_scale_std)
                        tmp.extend(rotation_std)
                        tmp.extend(transition_avg)
                        tmp.extend(transition_std)
                        tmp.extend(non_rigid_shape_parameters_avg)
                        tmp.extend(non_rigid_shape_parameters_std)
                        au_cr.append(tmp)

        except(TypeError):
                    print(TypeError)
                    print("Error in participant {0} and timestamp {1}.".format(point.participant_number,
                                                                               point.rate.timestamp))
    return au_cr, quadrant_labels, eng_labels, chal_labels, p_numbers
def calculate_labels(datapoint):
        #print("In set labels...")

        #           |
        #           |
        #    Ro=1   |   Fo=2
        # ----------|----------
        #           |
        #      Bo=3 |   Fr=4
        #           |

        margin_for_challenge=5
        margin_for_engagement=5
        if datapoint.rate.challenge> 50+margin_for_challenge:
            challenge_label = 1
            if datapoint.rate.engagement > 50+margin_for_engagement:
                quadrant_label = 2
                engagement_label = 1
            elif datapoint.rate.engagement < 50-margin_for_engagement:
                quadrant_label = 4
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'

        elif datapoint.rate.challenge < 50-margin_for_challenge:
            challenge_label = 0
            if datapoint.rate.engagement > 50+margin_for_engagement:
                quadrant_label = 1
                engagement_label = 1
            elif datapoint.rate.engagement < 50-margin_for_engagement:
                quadrant_label = 3
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'
        else:
            quadrant_label = "mr"
            challenge_label = 'mr'
            if datapoint.rate.engagement > 50+margin_for_engagement:
                engagement_label = 1
            elif datapoint.rate.engagement < 50-margin_for_engagement:
                engagement_label = 0
            else:
                quadrant_label = "mr"
                engagement_label = 'mr'

        if datapoint.rate.challenge == 0 or datapoint.rate.engagement == 0:
            quadrant_label = 'mr'
            engagement_label = 'mr'
            challenge_label = "mr"

        return quadrant_label, engagement_label, challenge_label
        #print("quad label is: ", self.quadrant_label)

def export(participants):
    data_points = []
    for k in range(len(participants)):
        for dp in participants[k].data_points:
            if dp.rate.engagement != 0 and dp.rate.challenge != 0:
                data_points.append(dp)
    au_cr, quadrant_labels, eng_labels, chal_labels, p_numbers = prepare_features(data_points)

    with open('au_cr.txt', 'w') as f:
        for item in au_cr:
            f.write("%s\n" % item)