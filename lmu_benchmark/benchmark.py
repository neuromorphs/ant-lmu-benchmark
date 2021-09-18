from lmu_benchmark import ldn
from lmu_benchmark import stmnist
import nengo
import numpy as np
import pytry
import sklearn.metrics

class Sinewaves(object):
    def __init__(self, freqs, stdev=0.1, n_samples=10, T=2.0):
        self.freqs = freqs
        self.stdev = stdev
        self.n_samples = n_samples
        self.T = T
    def generate(self, dt, rng):
        t = np.arange(int(self.T/dt))*dt
        dataset = []
        for i, f in enumerate(self.freqs):
            samples = []
            for j in range(self.n_samples):
                ff = f + rng.normal()*self.stdev
                samples.append(np.sin(t*2*np.pi*ff))
            dataset.append((np.array(samples), np.eye(len(self.freqs))[i]))
        return dataset

class LMUBenchmark(pytry.PlotTrial):
    def params(self):
        self.param('time step', dt=0.001)
        self.param('number of Legendre bases', q=6)
        self.param('memory window (in seconds)', theta=0.5)
        self.param('number of neurons', n_neurons=200)
        self.param('dimension of input signal', size_in=100)
        self.param('neuron type', neuron_type='nengo.LIF()')
        self.param('task to perform', task='Sinewaves([1,2])')
        self.param('output synapse time constant', output_synapse=0.01)
        self.param('proportion of dataset for training', p_training=0.8)

    def evaluate(self, p, plt):

        rng = np.random.RandomState(seed=p.seed)
        ldn_process = ldn.LDN(q=p.q, theta=p.theta, size_in=p.size_in)
        dataset = eval(p.task, globals(), locals()).generate(dt=p.dt, rng=rng)
        print(p.size_in)
        training_inputs = []
        training_outputs = []
        testing_inputs = []
        testing_outputs = []
        for inputs, category in dataset:
            order = np.arange(len(inputs),dtype=int)
            # rng.shuffle(order)
            N = int(len(order)*p.p_training)
            N1 = len(order)-N
            print("N=",N)
            print("N1=",N1)
            for i in range(N):
                training_inputs.append(inputs[i])
            for i in range(N1):
                testing_inputs.append(inputs[N+i])
            for input in inputs[0:N]:
                # print(np.array(input).shape[1])
                len_inp = np.array(input).T.shape[0]
                training_outputs.append(np.tile(category[None,:], (len_inp,1)))
            for input in inputs[N:len(order)]:
                len_inp = np.array(input).T.shape[0]
                testing_outputs.append(np.tile(category[None,:], (len_inp,1)))

        training_inputs = np.hstack(training_inputs).T
        testing_inputs = np.hstack(testing_inputs).T
        training_outputs = np.vstack(training_outputs)
        testing_outputs = np.vstack(testing_outputs)
        # training_inputs = np.array(training_inputs)
        # testing_inputs = np.array(testing_inputs)
        # training_outputs = np.array(training_outputs)
        # testing_outputs = np.array(testing_outputs)
        print(training_inputs.shape)
        print(testing_inputs.shape)
        print(training_outputs.shape)
        print(testing_outputs.shape)

        # print(training_inputs[:,None].shape)
        if p.size_in == 1:
            inputs = ldn_process.apply(training_inputs[:,None])
        else:
            inputs = ldn_process.apply(training_inputs[:])

        model = nengo.Network(seed=p.seed)
        with model:
            stim = nengo.Node(nengo.processes.PresentInput(testing_inputs, presentation_time=p.dt))
            ldn_node = nengo.Node(ldn_process)
            nengo.Connection(stim, ldn_node, synapse=None)
            neurons = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=ldn_node.size_out,
                                     neuron_type=eval(p.neuron_type))
            nengo.Connection(ldn_node, neurons, synapse=None)
            output = nengo.Node(None, size_in=training_outputs.shape[1])
            nengo.Connection(neurons, output, eval_points=inputs, function=training_outputs, synapse=None)
            p_output = nengo.Probe(output, synapse=p.output_synapse)
            p_stim = nengo.Probe(stim)

        sim = nengo.Simulator(model, progress_bar=False)
        with sim:
            sim.run(len(testing_inputs)*p.dt)

        if plt:
            plt.plot(sim.trange(), sim.data[p_stim])
            plt.plot(sim.trange(), sim.data[p_output])

        prediction = np.argmax(sim.data[p_output], axis=1)
        C = sklearn.metrics.confusion_matrix(np.argmax(testing_outputs, axis=1), prediction)

        accuracy = np.sum(np.diag(C))/np.sum(C)

        return dict(accuracy=accuracy, confusion=C.tolist())
