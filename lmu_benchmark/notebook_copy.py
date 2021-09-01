import matplotlib.pyplot as plt
import numpy as np
import pytry
import pandas
import lmu_benchmark
import seaborn

trial = lmu_benchmark.LMUBenchmark()
trial.run(task="Sinewaves([1,2])")

upper_freqs = [1.1, 1.2, 1.3, 1.4]
for seed in range(5):
    for upper in upper_freqs:
        for q in [1,3,6]:
            trial.run(q=q, seed=seed, task="Sinewaves([1,%g])"%upper,
            data_dir='exp3', verbose=False)

data = pytry.read('exp3')
df = pandas.DataFrame(data)
df

plt.figure(figsize=(12,4))
seaborn.barplot(x='task', y='accuracy', data=df, hue='q')
