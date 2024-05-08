from __future__ import annotations
from typing import Union

import numpy as np
import torch

from tqdm.autonotebook import trange
from ase import units

from ..graphs import GraphModule


class Variable:
    def __init__(
        self,
        name: str,
        values: dict[str, torch.Tensor],
        model_input_map: dict[str, torch.Tensor] = dict(),
        device: Union[str, torch.device] = None,
    ) -> None:
        """Tracks the state of a quantity (eg. position, cell, species, 
        volume) on each particle or each system in an MD simulation. Can
        also hold additional values associated to that quantity (such as
        velocity, acceleration, etc...)

        Parameters
        ----------
        name : str
            name for variable
        values : dict[str, torch.Tensor]
            dictionary of tracked values in the form `value_name: value`
        model_input_map : dict[str, torch.Tensor], optional
            dictionary of correspondences between values tracked by Variable
            and values input to HIP-NN model in the form 
            `hipnn-model-input-key: variable-value-key`, by default dict()
        device : Union[str, torch.device], optional
            device on which to keep values, by default None
        """        
        
        self.name = name
        self.model_input_map = model_input_map
        self.set_values(values)

        if device is not None:
            self.to(device)

    def set_values(self, values):
        if not isinstance(values, dict):
            raise TypeError("The argument for 'values' must be a dictionary.")

        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            if value.shape[0] != 1:  # GraphModule requires inputs in batches
                value.unsqueeze_(0)
            values[key] = value

        self.data = values

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Moves values stored in internal variable to 'device'.
        """
        self.device = device
        for key, value in self.data.items():
            self.data[key] = value.to(device)


class StaticVariable(Variable):
    '''
    A Variable child class for Variables whose values do not change
    during the MD simulation.
    '''
    pass


class DynamicVariable(Variable):
    '''
    A Variable child class for Variables whose values change
    during the MD simulation.
    '''
    def __init__(
        self,
        name: str,
        starting_values: dict[str, torch.Tensor],
        model_input_map: dict[str, str] = dict(),
        updater: DynamicVariableUpdater = None,
        device: Union[str, torch.device] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_input_map : dict[str, str]
            {hippynn_key: variable_key}
        starting_values :
        device : str or torch.device, optional
            device on which to keep values, by default None

        """
        self.name = name
        self.model_input_map = model_input_map

        self.set_updater(updater)
        self.set_values(starting_values)

        if device is not None:
            self.to(device)

        # # data must be put into model in batches, so we use batch size 1. May cause errors if only one particle.
        # for key, value in self.data.items():
        #     if value is not None:
        #         if value.shape[0] != 1:
        #             self.data[key] = value.unsqueeze(0)

    def set_updater(self, updater: DynamicVariableUpdater):
        self.updater = updater
        if updater is not None:
            updater._attach(self)


class DynamicVariableUpdater:
    def __init__(self):
        pass

    def _attach(self, variable: DynamicVariable):
        self.variable = variable

    def pre_step(self, dt):
        pass

    def post_step(self, dt, model_outputs):
        pass


class VelocityVerlet(DynamicVariableUpdater):
    def __init__(self, force_key, units_force = units.eV, units_acc = units.Ang/(1.0 ** 2)):
        self.force_key = force_key
        self.force_factor = units_force / units_acc

    def pre_step(self, dt):
        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + 0.5 * dt * self.variable.data["acceleration"]
        )
        self.variable.data["position"] = (
            self.variable.data["position"] + self.variable.data["velocity"] * dt
        )

    def post_step(self, dt, model_outputs):
        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)
        if len(self.variable.data["force"].shape) == len(
            self.variable.data["mass"].shape
        ):
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach()
                / self.variable.data["mass"]
                * self.force_factor
            )
        else:
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach()
                / self.variable.data["mass"][..., None]
                * self.force_factor
            )
        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + 0.5 * dt * self.variable.data["acceleration"]
        )


class LangevinDynamics(DynamicVariableUpdater):
    def __init__(self, force_key, temperature, frix, units_force = units.eV, units_acc = units.Ang/(1.0 ** 2)):
        self.force_key = force_key
        self.force_factor = units_force / units_acc
        self.temperature = temperature
        self.frix = frix
        self.kB = 0.001987204 * self.force_factor

        #TODO: allow to set seed

    def pre_step(self, dt):
        self.variable.data["position"] = (
            self.variable.data["position"] + self.variable.data["velocity"] * dt
        )

    def post_step(self, dt, model_outputs):
        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)

        self.variable.data["acceleration"] = (
            self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_factor
        )

        self.variable.data["velocity"] = self.variable.data["velocity"] + (
            +dt * self.variable.data["acceleration"]
            - self.frix * self.variable.data["velocity"] * dt
            + torch.sqrt(
                2
                * self.kB
                * self.frix
                * self.temperature
                / self.variable.data["mass"]
                * dt
            )
            * torch.randn_like(self.variable.data["velocity"])
        )


class MolecularDynamics:
    def __init__(
        self,
        dynamic_variables: list[DynamicVariable],
        static_variables: list[StaticVariable],
        model: GraphModule,
    ):
        self.dynamic_variables = dynamic_variables
        self.static_variables = static_variables
        self.model = model

        # keys = [f"output_{node.name}" for node in model.nodes_to_compute] + [
        #     f"{variable.name}_{key}"
        #     for variable in [*self.dynamic_variables, *self.static_variables]
        #     for key in variable.data.keys()
        # ]
        # self.data = {key: [] for key in keys}
        self._data = dict()

    def _step(
        self,
        dt,
    ):
        for variable in self.dynamic_variables:
            variable.updater.pre_step(dt)

        model_outputs = self.model(
            **{
                hippynn_key: variable.data[variable_key]
                for variable in [*self.dynamic_variables, *self.static_variables]
                for hippynn_key, variable_key in variable.model_input_map.items()
            }
        )

        for variable in self.dynamic_variables:
            variable.updater.post_step(dt, model_outputs)

        return model_outputs
    
    def _update_data(self, model_outputs):
        for variable in [*self.dynamic_variables, *self.static_variables]:
            for key, value in variable.data.items():
                try:
                    self._data[f"{variable.name}_{key}"].append(value.cpu().detach()[0])
                except KeyError:
                    self._data[f"{variable.name}_{key}"] = [value.cpu().detach()[0]]
        for key, value in model_outputs.items():
            try:
                self._data[f"output_{key}"].append(value.cpu().detach()[0])
            except KeyError:
                self._data[f"output_{key}"] = [value.cpu().detach()[0]]

    def run(self, dt, n_steps, record_every: int = None):
        for i in trange(n_steps):
            model_outputs = self._step(dt)
            if record_every is not None and (i + 1) % record_every == 0:
                self._update_data(model_outputs)
        pass

    def get_data(self):
        '''Returns a dictionary of the recorded data'''
        return {key: np.array(value) for key, value in self._data.items()}
    
    def reset_data(self):
        '''Clear all recorded data'''
        self._data = {key: [] for key in self._data.keys()}

