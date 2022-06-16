import os
import re

import numpy as np

from park.envs.circuit.simulator.circuit import Circuit, export_circuit
from park.envs.circuit.simulator.utility.io import load_txt
from park.envs.circuit.simulator.utility.misc import AttrDict
from park.spaces import Box

__all__ = ['ThreeStageTranimpedenceAmplifier']


@export_circuit
class ThreeStageTranimpedenceAmplifier(Circuit):
    @property
    def simdata(self):
        if not hasattr(self, '__cache_simdata__'):
            data = load_txt('./park/envs/circuit/simulator/library/transimpedance/three_stage.circuit')
            setattr(self, '__cache_simdata__', data)
        return getattr(self, '__cache_simdata__')

    @property
    def libdata(self):
        if not hasattr(self, '__cache_libdata__'):
            data = load_txt('./park/envs/circuit/simulator/library/ee114.hspice')
            setattr(self, '__cache_libdata__', data)
        return getattr(self, '__cache_libdata__')

    @property
    def parameters(self):
        return 'W1', 'WL1', 'WB1', 'W2', 'WL2', 'WB2', 'W3', 'WB3', 'WB', 'L1', \
               'LL1', 'LB1', 'L2', 'LL2', 'LB2', 'L3', 'LB3', 'LB', 'RB'

    @classmethod
    def _get_transistor_states(cls, result):
        states = AttrDict(
            M1a=[1, 0], M1la=[0, 1], Mb1a=[1, 0], M1b=[1, 0], M1lb=[0, 1],
            Mb1b=[1, 0], M2a=[1, 0], M2la=[0, 1], Mb2a=[1, 0], M2b=[1, 0],
            M2lb=[0, 1], Mb2b=[1, 0], M3a=[1, 0], Mb3a=[1, 0], M3b=[1, 0],
            Mb3b=[1, 0], Mb=[1, 0]
        )

        regex = re.compile(r'.*[\s\w]+\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+'
                           r'([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+).*')
        for line in result[82: 102]:
            data = regex.findall(line)[0]
            states.M1a.append(cls.number_from_string(data[0]))
            states.M1la.append(cls.number_from_string(data[1]))
            states.Mb1a.append(cls.number_from_string(data[2]))
            states.M1b.append(cls.number_from_string(data[3]))
            states.M1lb.append(cls.number_from_string(data[4]))
            states.Mb1b.append(cls.number_from_string(data[5]))

        for line in result[109: 129]:
            data = regex.findall(line)[0]
            states.M2a.append(cls.number_from_string(data[0]))
            states.M2la.append(cls.number_from_string(data[1]))
            states.Mb2a.append(cls.number_from_string(data[2]))
            states.M2b.append(cls.number_from_string(data[3]))
            states.M2lb.append(cls.number_from_string(data[4]))
            states.Mb2b.append(cls.number_from_string(data[5]))

        regex = re.compile(r'.*[\s\w]+\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+'
                           r'([+\-.\d\w]+)\s+([+\-.\d\w]+).*')
        for line in result[136: 156]:
            data = regex.findall(line)[0]
            states.M3a.append(cls.number_from_string(data[0]))
            states.Mb3a.append(cls.number_from_string(data[1]))
            states.M3b.append(cls.number_from_string(data[2]))
            states.Mb3b.append(cls.number_from_string(data[3]))
            states.Mb.append(cls.number_from_string(data[4]))
        return states

    @classmethod
    def _get_curve(cls, result):
        frequency, magnitude, phase = [], [], []
        regex = re.compile(r'.*\s+([\d\w.+]+)\s+([\d\w.+\-]+)\s+([\d\w.+\-]+).*')
        for line in result[167: 8168]:
            curve = regex.findall(line)[0]
            frequency.append(cls.number_from_string(curve[0]))
            magnitude.append(cls.number_from_string(curve[1]))
            phase.append(cls.number_from_string(curve[2]))
        return AttrDict(frequency=frequency, magnitude=magnitude, phase=phase)

    @classmethod
    def _get_metrics(cls, values, result):
        proto_gain = re.findall(r'.*gainmax=([+\-.\d\s\w]+)at', result[8169])[0]
        proto_bandwidth = re.findall(r'.*f3db=([+\-.\d\s\w]+).*', result[8171])[0].strip()
        metrics = AttrDict(
            power=cls.number_from_string(re.findall(r'.*=([+\-.\d\s\w]+)watts.*', result[49])[0]),
            gain=10 ** (cls.number_from_string(proto_gain) / 20) * 2,
            bandwidth=100e6 if proto_bandwidth == 'failed' else cls.number_from_string(proto_bandwidth)
        )
        metrics.area = 0
        metrics.area += 2 * (values.W1 * values.L1 + values.WL1 * values.LL1 + values.WB1 * values.LB1)
        metrics.area += 2 * (values.W2 * values.L2 + values.WL2 * values.LL2 + values.WB2 * values.LB2)
        metrics.area += 2 * (values.W3 * values.L3 + values.WB3 * values.LB3) + values.WB * values.LB

        metrics.saturated = all([
            re.match(r'.*region\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati.*', result[81]),
            re.match(r'.*region\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati.*', result[108]),
            re.match(r'.*region\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati.*', result[135])
        ])
        return metrics

    @classmethod
    def _get_voltages(cls, auxiliaries):
        regex = re.compile(r'.*=\s*([-+\d]+\.\d*\w*)')
        voltages = []
        for line in auxiliaries[12: 24]:
            voltages.append(cls.number_from_string(regex.search(line).group(1)))
        return voltages

    def run(self, tmp_path, values):
        with open(os.path.join(tmp_path, 'three_amp.sp'), 'w') as writer:
            data = self.simdata
            args_units = ['u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u',
                          'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'k']
            for key, tail in zip(self.parameters, args_units):
                data = data.replace(f'$(params:{key})', f'%f{tail}' % getattr(values, key))
            writer.write(data)
        with open(os.path.join(tmp_path, 'ee114.hspice'), 'w') as writer:
            parameters = '.param L1=%fu, L2=%fu, L3=%fu, L4=%fu, L5=%fu, L6=%fu, L7=%fu, L8=%fu, L9=%fu' % (
                values.L1, values.LL1, values.LB1, values.L2, values.LL2, values.LB2, values.L3, values.LB3, values.LB)
            writer.write(self.libdata.replace('$(params:all_L)', parameters))

        result = self._run_hspice('three_amp', tmp_path).split('\n')
        with open(os.path.join(tmp_path, 'three_amp.ic0'), 'r') as reader:
            auxiliaries = reader.read().split('\n')
        transistor_states = self._get_transistor_states(result)
        curve = self._get_curve(result)
        metrics = self._get_metrics(values, result)
        voltages = self._get_voltages(auxiliaries)
        return AttrDict(
            metrics=metrics,
            curve=curve,
            transistor_states=transistor_states,
            voltages=voltages,
            tmp_path=os.path.abspath(tmp_path)
        )

    @property
    def out_space(self):
        return AttrDict(
            metrics=AttrDict(
                power=Box(low=-np.inf, high=np.inf, shape=[], dtype=np.float32),
                gain=Box(low=-np.inf, high=np.inf, shape=[], dtype=np.float32),
                bandwidth=Box(low=-np.inf, high=np.inf, shape=[], dtype=np.float32),
                area=Box(low=-np.inf, high=np.inf, shape=[], dtype=np.float32),
                saturated=Box(shape=[], low=0, high=1, dtype=np.float32) # bool
            ),
            transistor_states=Box(low=-np.inf, high=np.inf, shape=[17], dtype=np.float32),
            voltages=Box(low=-np.inf, high=np.inf, shape=[12], dtype=np.float32),
            curve=AttrDict(
                frequency=Box(low=-np.inf, high=np.inf, shape=[8001], dtype=np.float32),
                magnitude=Box(low=-np.inf, high=np.inf, shape=[8001], dtype=np.float32),
                phase=Box(low=-np.inf, high=np.inf, shape=[8001], dtype=np.float32)
            )
        )
