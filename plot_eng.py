import matplotlib.pyplot as plt
#participants =[]
eng = []

for participant in participants:
    for dp in participant.data_points:
        eng.append(dp.rate.engagement)



fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(eng, 20, density=True, histtype='step',
                           cumulative=True, label='Empirical')
plt.show()