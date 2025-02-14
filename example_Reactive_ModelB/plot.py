import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
data_integral = np.loadtxt('Integral.out', delimiter=' ')
data_active = np.loadtxt('Wakeup_portion.out', delimiter=' ')

t1 = data_integral[:, 0]
x = data_integral[:, 1]
y = data_integral[:, 2]
z = data_integral[:, 3]
b = data_integral[:, 4]

err = np.abs(y - y[0]) / np.abs(y[0])

t2 = data_active[:, 0]
active = data_active[:, 1]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
#ax.plot(x, y)
#ax.plot(t1, err, label='Error in integral of c')
ax.plot(t1, x, label='Integral of x')
ax.plot(t1, y, label='Integral of y')
ax.plot(t1, z, label='Integral of z')
ax.plot(t1, b, label='Integral of b')
ax.plot(t2, active, label='Active portion of the system')

# Set the axis labels and title
ax.set_xlabel('t')
ax.set_ylabel('Error in integral of c')
#ax.set_ylabel('Integral of c')
#ax.set_title('Plot of X vs Y')

# Save the figure
plt.legend()
plt.savefig('plot_integrals.png', dpi=300)