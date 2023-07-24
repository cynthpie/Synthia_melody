import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("sine_outputs_sb_0.0.txt")

fig1, ax1 = plt.subplots()
ax1.boxplot(data)
plt.show()
plt.savefig('foo.png')
# plt.close(fig) 