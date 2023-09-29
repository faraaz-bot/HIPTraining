import matplotlib.pyplot as plt
import numpy as np
data0 = np.genfromtxt('raw_signal.dat')
plt.plot(data0, label='Raw Signal')
plt.xlabel('x')
plt.ylabel('Signal')
#
# Add more plots to visualize your moving filtered signal
#
#
#
plt.legend()
plt.show()
