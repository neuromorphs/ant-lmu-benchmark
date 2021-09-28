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

# trial1 = lmu_benchmark.LMUBenchmark()
# trial1.run(size_in=100,
# task='stmnist.STMNIST([1,2,3,4],500)')

trial2 = lmu_benchmark.LMUBenchmark()
trial2.run(size_in=100,
task='stmnist.STMNIST([1,2],500)')
