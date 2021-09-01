import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import nengo
import pytry
import seaborn
from lmu_benchmark import stmnist
import lmu_benchmark


stm = stmnist.STMNIST([1,2],5)
# 100 spiketrains per sample stored as:
# training_data[sample][spiketrain_i] i=0,1,..,99
training_data=stm.generate(0)
print(training_data)

trial = lmu_benchmark.LMUBenchmark()
trial.run(size_in=100,task='stmnist.STMNIST([1,2],5)')

for i in range(100):
    plt.plot(training_data[1][i])
plt.show()
