from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

df = pd.read_csv('/home/nlsde/RLmodel/Double-DQN/result2.csv')
internal = df.iloc[:, 0]
blocksize = df.iloc[:, 1]
tps = df.iloc[:, 2]

yi = np.linspace(min(blocksize), max(blocksize))
xi = np.linspace(min(internal), max(internal))
xi, yi = np.meshgrid(xi, yi)
zi = griddata(df.iloc[:, 0:2], tps, (xi, yi), method='cubic')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
# ax2 = fig.gca(projection='3d')
surf=ax2.plot_surface(xi,yi,zi,cmap='BuPu',linewidth=0,antialiased=False)
fig.colorbar(surf)
ax2.set_title('tps_data_trend')
plt.savefig('./my.png')
# ax2.plot_surface(X, Y, TPS, cmap='rainbow')
# plt.show()
