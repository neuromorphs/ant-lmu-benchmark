import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import nengo
import pytry
from lmu_benchmark import stmnist
import lmu_benchmark

# 100 spiketrains per sample stored as:
# training_data[sample][spiketrain_i] i=0,1,..,99
# stm = stmnist.STMNIST([1,2,3],5)
# training_data=stm.generate(0)
#
# td = np.array(training_data,dtype=object)
# print(td.shape)
# td1=td[0,0]
# print(np.array(td1[3]).shape)
#
# for inputs, category in training_data:
#     print(inputs[1])

trial = lmu_benchmark.LMUBenchmark()
trial.run(size_in=100,
task='stmnist.STMNIST([1,2],5)')
