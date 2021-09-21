import scipy.io
import numpy as np


class STMNIST(object):
    def __init__(self, digits=np.arange(10), samples_per_digit=20):
        self.digits = digits
        self.samples_per_digit = samples_per_digit

    def generate(self, rng, dt=0.001):
        dataset = []
        i = 0
        for digit in self.digits:
            samples = []
            for sample in range(self.samples_per_digit):
                samples.append(self.load_digit(digit, sample, dt))
            dataset.append((samples, np.eye(len(self.digits))[i]))
            i = i + 1
        return dataset

    def load_digit(self, digit, n, dt):
        # Change this to your dataset path
        dataset_path = 'C:/Users/karth/Desktop/Telluride/ST-MNIST/'
        path = dataset_path+'data_submission/'+str(
                digit)+'/'+str(digit)+'_ch0_'+str(n)+'_spiketrain.mat'
        mat = scipy.io.loadmat(path)
        mdata = mat['spiketrain']
        mdata = np.array(mdata)
        spikes = mdata[0:100, :]
        times = mdata[100, :]
        spikedata = np.zeros([100, int(times[-1]/dt)+1])
        # times1=np.linspace(0,int(times[-1]/dt),int(times[-1]/dt)+1)
        for t in range(len(times)):
            spikedata[:, int(times[t]/dt)] = spikes[:, t]
        return spikedata
