from benchmark import Sinewaves
import numpy as np
import matplotlib.pyplot as plt

s1 = Sinewaves([1,2])
rng = np.random.RandomState()
s1_data = s1.generate(0.001,rng)
ss = np.array(s1_data,dtype=object)
print(ss.shape)
print(ss)
ss1 = ss[0,0]

# plt.subplot(2,1,1)
# for i in range(10):
#         plt.plot(ss1[i,:])
# ss2 = ss[0,1,0]
# plt.subplot(2,1,2)
# for i in range(10):
#         plt.plot(ss2[i,:])
# plt.show()
