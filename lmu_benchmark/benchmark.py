from lmu_benchmark import ldn
import nengo
import numpy as np
import pytry

class LMUBenchmark(pytry.PlotTrial):
    def params(self):
        self.param('time step', dt=0.001)
        self.param('number of Legendre bases', q=6)
        self.param('memory window (in seconds)', theta=0.5)
        self.param('number of neurons', n_neurons=200)
        self.param('neuron type', neuron_type='nengo.LIF()')
        self.param('task to perform', task='sinewaves([1,2])')
        self.param('output synapse time constant', output_synapse=0.01)

    def evaluate(self, p, plt):


        def sinewaves(freqs):
            T = p.theta*10
            t = np.arange(int(T/p.dt))*p.dt
            inputs = []
            outputs = []
            for i, f in enumerate(freqs):
                inputs.append(np.sin(t*2*np.pi*f))
                outputs.append(np.tile(np.eye(len(freqs))[i:i+1], (len(t), 1)))
            return np.hstack(inputs), np.vstack(outputs)

        ldn_process = ldn.LDN(q=p.q, theta=p.theta)
        data_inputs, data_outputs = eval(p.task, locals())

        inputs = ldn_process.apply(data_inputs[:,None])

        model = nengo.Network(seed=p.seed)
        with model:
            stim = nengo.Node(nengo.processes.PresentInput(data_inputs, presentation_time=p.dt))
            ldn_node = nengo.Node(ldn_process)
            nengo.Connection(stim, ldn_node, synapse=None)
            neurons = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=ldn_node.size_out,
                                     neuron_type=eval(p.neuron_type))
            nengo.Connection(ldn_node, neurons, synapse=None)
            output = nengo.Node(None, size_in=data_outputs.shape[1])
            nengo.Connection(neurons, output, eval_points=inputs, function=data_outputs, synapse=None)
            p_output = nengo.Probe(output, synapse=p.output_synapse)

        sim = nengo.Simulator(model, progress_bar=False)
        with sim:
            sim.run(len(data_inputs)*p.dt)

        if plt:
            plt.plot(sim.trange(), sim.data[p_output])



        





