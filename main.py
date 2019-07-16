from participant import Participant
import pickle


from openface_preprocessing import Openface_Preprocessing


# This function set the participants objects
def set_participants():
    participants = []

    ##p5 = Participant(5, "15-3-2019", 26, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\5")
    ##p6 = Participant(6, "18-3-2019", 33, 'm', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\6")
    ##p7 = Participant(7, "20-3-2019", 31, 'm', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\7")
    ##p8 = Participant(8, "21-3-2019", 33, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\8")
    ##p9 = Participant(9, "29-3-2019", 32, 'm', "Pallete Logs\\9")
    ##p10 = Participant(10, "1-4-2019", 27, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\10")
    ##p11 = Participant(11, "2-4-2019", 28, 'm', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\11")
    ##p12 = Participant(12, "4-3-2019", 27, 'm', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\12")
    ##p14 = Participant(14, "9-4-2019", 27, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\14")
    p15 = Participant(15, "15-4-2019", 32, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\15")
    ##p16 = Participant(16, "16-4-2019", 34, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\16")
    ##p17 = Participant(17, "17-5-2019", 34, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\17")
    ##p18 = Participant(18, "10-5-2019", 27, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\18")
    ##p19 = Participant(19, "14-5-2019", 33, 'f', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\19")
    ##p20 = Participant(20, "19-5-2019", 32, 'm', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\20")

    ##p5.set_path_of_participant_snapshots("C:\Webcam Snapshots\\5\\5")
    ##p6.set_path_of_participant_snapshots("C:\Webcam Snapshots\\6\\6")
    ##p7.set_path_of_participant_snapshots("C:\Webcam Snapshots\\7\\7\\7")
    ##p8.set_path_of_participant_snapshots("C:\Webcam Snapshots\\8")
    ##p9.set_path_of_participant_snapshots("C:\Webcam Snapshots\\9")
    ##p10.set_path_of_participant_snapshots("C:\Webcam Snapshots\\10\\10\\10")
    ##p11.set_path_of_participant_snapshots("C:\Webcam Snapshots\\11\\11\\11")
    ##p12.set_path_of_participant_snapshots("C:\Webcam Snapshots\\12\\12")
    ##p14.set_path_of_participant_snapshots("C:\Webcam Snapshots\\14\\14")
    p15.set_path_of_participant_snapshots("C:\Webcam Snapshots\\15\\15")
    ##p16.set_path_of_participant_snapshots("C:\Webcam Snapshots\\16")
    ##p17.set_path_of_participant_snapshots("C:\Webcam Snapshots\\17")
    ##p18.set_path_of_participant_snapshots("C:\Webcam Snapshots\\18\\18")
    ##p19.set_path_of_participant_snapshots("C:\Webcam Snapshots\\19\\19")
    ##p20.set_path_of_participant_snapshots("C:\Webcam Snapshots\\20")

    #participants.extend([p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
    ##participants.append(p5)
    ##participants.append(p6)
    ##participants.append(p7)
    ##participants.append(p8)
    ##participants.append(p9)
    ##participants.append(p10)
    ##participants.append(p11)
    ##participants.append(p12)
    ##participants.append(p14)
    participants.append(p15)
    ##participants.append(p16)
    ##participants.append(p17)
    ##participants.append(p18)
    ##participants.append(p19)
    ##participants.append(p20)

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
        try:
            print("### Starting Analysis on participant number "+str(participant.number)+" ...")

            participant.preparation_for_facial_expression_analysis(period=195, margin=15,
            path_for_saving_datapoint_frames = "Y:\\Openface_Processed_Frames\\Folders_of_Datapoints_Frames")

            #participant.preparation_for_facial_expression_analysis_with_rar_file(period=195, margin=15,
            #path_for_saving_datapoint_frames = "C:\\Folders_of_Datapoints_Frames")

            #for datapoint in participant.data_points:
            #   print(datapoint.openface_object.to_string())

            filehandler = open("Y:\\Openface_Processed_Frames\\Participant_objects\\" + str(participant.number) + ".obj", 'wb')
            pickle.dump(participant, filehandler)
        except OSError as e:
            print(e)



if __name__ == "__main__":
    main()
