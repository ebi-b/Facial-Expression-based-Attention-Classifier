from participant import Participant

participants = []

p5 = Participant(5, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\5")
p6 = Participant(6, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\6")
p7 = Participant(7, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\7")
p8 = Participant(8, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\8")
p9 = Participant(9, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\9")
p10 = Participant(10, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\10")
p11 = Participant(11, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\11")
p12 = Participant(12, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\12")
p13 = Participant(13, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\13")
p14 = Participant(14, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\14")
p15 = Participant(15, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\15")
p16 = Participant(16, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\16")
p17 = Participant(17, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\17")
p18 = Participant(18, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\18")
p19 = Participant(19, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\19")
p20 = Participant(20, "", 25, 'M', "Y:\Git\Facial Expression Analysis\Analysis_on_Dataset\Pallete Logs\\20")

participants.extend([p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])


# Checking if Number of Rate is ok.
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
