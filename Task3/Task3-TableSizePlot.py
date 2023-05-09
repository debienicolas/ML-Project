import matplotlib.pyplot as plt

# data
x_values = ['1x1', '1x2', '1x3', '1x4', '2x2', '2x3']
y1_values = [640, 4696, 36960, 295000, 147552, 5242968]
y2_values = [360, 2272, 9312, 73816, 18520, 1310808]
y3_values = [15, 127, 1023, 8191, 4095, 131071]
y4_values = [8, 47, 311, 2239, 569, 33407]

# create figure and axis objects with twin y-axis
fig, ax1 = plt.subplots()

# set logarithmic scales
ax1.set_yscale('log')
ax1.set_xscale('linear')

# create first y-axis with red color and star marker
color1 = 'orange'
color2 = 'purple'
ax1.set_ylabel('Table size (bytes)', color=color1)
ax1.plot(x_values, y1_values, color=color1, marker='*', label='Transposition table size without symmetries')
ax1.plot(x_values, y2_values, color=color1, marker='o', label='Transposition table size with symmetries')
ax1.tick_params(axis='y', labelcolor=color1)

# create second y-axis with black color and circle marker
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Nr. of keys', color=color2)
ax2.plot(x_values, y3_values, color=color2, marker='*', label='Nr of keys without symmetries')
ax2.plot(x_values, y4_values, color=color2, marker='o', label='Nr of keys with symmetries')
ax2.tick_params(axis='y', labelcolor=color2)

# set logarithmic scales for second y-axis
ax2.set_yscale('log')

# set title and labels
ax1.set_title('Tiny Dots-and-Boxes average transposition table size')
ax1.set_xlabel('Game size')

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
plt.legend(lines, labels)

# save and show plot
plt.savefig('Task3-AvgTableSize.pdf')
plt.show()
