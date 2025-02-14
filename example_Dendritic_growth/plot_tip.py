import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
data = np.genfromtxt('tip_profile.out')

# Extract the x-coordinates and y-coordinates
x = np.arange(len(data[0]))
y = data
print(len(y[0]))

# Plot the data
plt.figure(figsize=(12, 8))
#for i in range(len(y)):
for i in range(5, 11):
    plt.loglog(x, y[i]-y[i][0], label=f'Interface {i+1}')

# Plot a line with slope=1.5
y_ref = 1.5 * x
plt.loglog(x, y_ref, '--r', label='Slope=1.5')

plt.xlabel('X-coordinates')
plt.ylabel('Z-coordinates')
plt.title('Tip Profile')
plt.legend()
plt.grid()
plt.show()
