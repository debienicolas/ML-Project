import matplotlib.pyplot as plt
import numpy as np

# Define x and y values
x_values = ['1x1', '1x2', '1x3', '1x4', '2x2', '2x3']
y1_values = [0.000108445, 0.026857078, 13.49236149, np.nan, np.nan, np.nan]
y2_values = [8.84056E-05, 0.006488931, 0.078116608, 13.00671818, 0.829023182, np.nan]
y3_values = [0.000316525, 0.004059684, 0.061838162, 0.456027973, 0.241993213, 12.03562438]
y4_values = [0.000178409, 0.002095842, 0.028485751, 0.175082278, 0.076801539, 4.059689093]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x_values, y1_values, color='blue', marker='x', label='Minimax')
ax.plot(x_values, y2_values, color='green', marker='x', label='Minimax with AlphaBeta pruning')
ax.plot(x_values, y3_values, color='red', marker='x', label='Minimax with transposition table')
ax.plot(x_values, y4_values, color='black', marker='x', label='Minimax exploiting symmetries')

# Set y-axis to logarithmic
ax.set_yscale('log')
ax.set_ylabel('Average execution time (seconds)')

# Set x-axis label and tick labels
ax.set_xlabel('Game size')
ax.set_xticklabels(x_values)
ax.tick_params(axis='x', rotation=45)

# Set the title and legend
ax.set_title('Tiny Dots-and-Boxes average execution time')
ax.legend()

# Save the plot as a PDF
plt.savefig('Task3-AvgExeTimes.pdf')
plt.show()