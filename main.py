from participant import Participant
import pickle
from openface_preprocessing import Openface_Preprocessing


# This function set the participants objects
def set_participants():
    participants = []

    p9 = Participant(9, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\9")
    #p6 = Participant(6, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\6")
    #p7 = Participant(7, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\7")
    #p8 = Participant(8, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\8")
    #p9 = Participant(9, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\9")
    #p10 = Participant(10, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\10")
    #p11 = Participant(11, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\11")
    #p13 = Participant(13, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\13")
    #p14 = Participant(14, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\14")
    #p15 = Participant(15, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\15")
    #p16 = Participant(16, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\16")
    #p17 = Participant(17, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\17")
    #p18 = Participant(18, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\18")
    #p19 = Participant(19, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\19")
    #p20 = Participant(20, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\20")

    p9.set_path_of_participant_snapshots("C:\\Webcam Snapshots\\webcamSnapshot-9\\webcamSnapshot")
    #p6.set_path_of_snapshots()
    #p7.set_path_of_snapshots()
    #p8.set_path_of_snapshots()
    #p9.set_path_of_snapshots()
    #p10.set_path_of_snapshots()
    #p11.set_path_of_snapshots()
    #p13.set_path_of_snapshots()
    #p14.set_path_of_snapshots()
    #p15.set_path_of_snapshots()
    #p16.set_path_of_snapshots()
    #p17.set_path_of_snapshots()
    #p18.set_path_of_snapshots()
    #p19.set_path_of_snapshots()
    #p20.set_path_of_snapshots()

    #participants.extend([p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
    participants.append(p9)
    return participants


# Checking if Number of Rate is ok.
def check_rates(participants):
    count = 0
    p_number = 5
    for participant in participants:

        p_count = 0
        for rate in participant.rates:
            rate.print_rate()
            p_count += 1
            count += 1
        print("Number of Rates for Participant {0} is: {1}".format(p_number, p_count))
        p_number += 1
    print("Number of Rates in Total is: {0}".format(count))


# main function of the script
def main():
    participants = set_participants()
    for participant in participants:
        print("### Starting Analysis on participant number "+str(participant.number)+" ...")
        participant.preparation_for_facial_expression_analysis(period=195, margin=15,
          path_for_saving_datapoint_frames = "Y:\\Openface_Processed_Frames\\Folders_of_Datapoints_Frames")

        #for datapoint in participant.data_points:
        #   print(datapoint.openface_object.to_string())

    filehandler = open("Y:\\Openface_Processed_Frames\\Participant_objects\\" + str(participant.number) + ".obj", 'wb')
    pickle.dump(participant, filehandler)


if __name__== "__main__":
    main()