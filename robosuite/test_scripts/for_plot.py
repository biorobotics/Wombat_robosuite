import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
# x = np.array([0.20,0.25,0.3,0.35,0.40,0.45,0.50,0.55,0.6])
# y = np.array([100,0,100,0,100,100,100,0,100])

# x1 = np.array([0.20,0.25,0.3,0.35,0.40,0.45,0.50,0.55,0.6])
# y1 = np.array([0,0,0,0,0,0,0,0,0])

# x = np.array([0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75])
# y = np.array([68,0,4,0,0,24,72,76])
# x = np.array([0.4,0.4425,0.485,0.5275,0.57,0.6125,0.655,0.6975,0.74])
# y = np.array([72,96,96,96,100,96,92,84,96])
x = np.array([0.4,0.4425,0.485,0.5275,0.57,0.6125,0.655,0.6975,0.74])
y = np.array([0,4,0,8,60,15,8,0,4])
# Plot the data
plt.plot(x, y, label='Gripping Success rate')
# plt.plot(x1, y1, linestyle='dashed',label='earlier')
# Add a legend
plt.xlabel('iPhone x-coordinate (across the conveyor belt)')
plt.ylabel('gripping success rate in % (based on 25 runs)')
plt.ylim([-5,105])
plt.title('Gripping Success rate vs. iPhone x-coordinate')
plt.legend()

# Show the plot
plt.show()