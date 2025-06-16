'''
This module is only available if the `ase` package is installed.
'''

from __future__ import annotations
from typing import Optional
from functools import singledispatchmethod

import numpy as np
import torch
import ase

from ..tools import progress_bar
from ..graphs import Predictor
from ..layers.pairs.periodic import wrap_systems_torch


class Variable:
    """
    Tracks the state of a quantity (eg. position, cell, species,
    volume) on each particle or each system in an MD simulation. Can
    also hold additional data associated to that quantity (such as its
    velocity, acceleration, etc...)
    """

    def __init__(
        self,
        name: str,
        data: dict[str, torch.Tensor],
        model_input_map: dict[str, str] = dict(),
        updater: VariableUpdater = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param name: name for variable
        :param data: dictionary of tracked data in the form `value_name: value`
        :param model_input_map: dictionary of correspondences between data tracked by Variable
            and inputs to the HIP-NN model in the form
            `hipnn-db_name: variable-data-key`, defaults to dict()
        :param updater: object which will update the data of the Variable
            over the course of the MD simulation, defaults to None
        :param device: device on which to keep data, defaults to None
        :param dtype: dtype for float type data, defaults to None
        """
        self.name = name
        self.data = data
        self.model_input_map = model_input_map
        self.updater = updater
        self.device = device
        self.dtype = dtype

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.as_tensor(value)
           
        batch_sizes = set([value.shape[0] for value in data.values()])
        if len(batch_sizes) > 1:
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_sizes}. The first axis of each array in 'data' must be a batch axis of the same size."
            )

        self._data = data

    @property
    def model_input_map(self):
        return self._model_input_map

    @model_input_map.setter
    def model_input_map(self, model_input_map):
        for key, value in model_input_map.items():
            if not value in self.data.keys():
                raise ValueError(
                    f"Each value in the 'model_input_map' dictionary should correspond to a key in the 'data' dictionary. "
                    + f"Each key of the 'model_input_map' should correspond to hippynn db_name. Value {value} found in 'model_input_map', but no corresponding key in the 'data' dictionary found."
                )
        self._model_input_map = model_input_map

    @property
    def updater(self):
        return self._updater

    @updater.setter
    def updater(self, updater):
        if updater is None:
            self._updater = None
            return
        updater.variable = self
        self._updater = updater

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device is None:
            self._device = None
            return
        self._device = device
        for key, value in self.data.items():
            self.data[key] = value.to(device)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype is None:
            self._dtype = None
            return
        self._dtype = dtype
        float_dtypes = [torch.float, torch.float16, torch.float32, torch.float64]
        for key, value in self.data.items():
            if value.dtype in float_dtypes:
                self.data[key] = value.to(dtype)

    @singledispatchmethod
    def to(self, arg):
        raise ValueError(f"Argument must be of type torch.device or torch.dtype, but provided argument is of type {type(arg)}.")

    @to.register
    def _(self, arg: torch.device):
        self.device = arg

    @to.register
    def _(self, arg: torch.dtype):
        self.dtype = arg


class VariableUpdater:
    """
    Parent class for algorithms that make updates to the data of a Variable during each step of an MD simulation.

    Subclasses should redefine __init__, pre_step, post_step, and required_variable_data as needed. The inputs to pre_step and post_step should not be changed.
    """

    # A list of keys which must appear in Variable.data for any Variable that will be updated by objects of this class.
    # Checked for by variable.setter.
    required_variable_data = []

    def __init__(self):
        pass

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, variable):
        for key in self.required_variable_data:
            if key not in variable.data.keys():
                raise ValueError(
                    f"Cannot attach to Variable with no assigned values for {key}. Update Variable.data to include values for {key}."
                )
        self._variable = variable

    def pre_step(self, dt: float):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        """        
        pass

    def post_step(self, dt: float, model_outputs: dict):
        """Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :param model_outputs: dictionary of HIPNN model outputs
        """
        pass


class NullUpdater(VariableUpdater):
    """
    Makes no change to the variable data at each step of MD.
    """

    def pre_step(self, dt):
        pass

    def post_step(self, dt, model_outputs):
        pass

class VelocityVerlet(VariableUpdater):
    """
    Implements the Velocity Verlet algorithm
    """

    required_variable_data = ["position", "velocity", "acceleration", "mass"]

    def __init__(
        self,
        force_db_name: str,
        force_units: Optional[float] = None,
        position_units: Optional[float] = None,
        time_units: Optional[float] = None,
    ):
        """
        :param force_db_name: key which will correspond to the force on the corresponding Variable
            in the HIPNN model output dictionary
        :param force_units: model force units output (in terms of ase.units), defaults to eV/Ang
        :param position_units: model position units output (in terms of ase.units), defaults to Ang
        :param time_units: model time units output (in terms of ase.units), defaults to fs
        """
        self.force_key = force_db_name
        self.force_units = (force_units or ase.units.eV/ase.units.Ang)
        self.position_units = (position_units or ase.units.Ang)
        self.time_units = (time_units or ase.units.fs)

    def pre_step(self, dt: float):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        """
        self.variable.data["velocity"] = self.variable.data["velocity"] + 0.5 * dt * self.variable.data["acceleration"]
        self.variable.data["position"] = self.variable.data["position"] + self.variable.data["velocity"] * dt
        
        if "cell" in self.variable.data.keys():
            _, self.variable.data["position"], *_ = wrap_systems_torch(coords=self.variable.data["position"], cell=self.variable.data["cell"], cutoff=0) # cutoff only impacts unused outputs; can be set arbitrarily
            try:
                self.variable.data["unwrapped_position"] = self.variable.data["unwrapped_position"] + self.variable.data["velocity"] * dt
            except KeyError:
                self.variable.data["unwrapped_position"] = self.variable.data["position"].clone().detach()

    def post_step(self, dt: float, model_outputs: dict):
        """Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :param model_outputs: dictionary of HIPNN model outputs
        """
        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)
        if len(self.variable.data["force"].shape) == len(self.variable.data["mass"].shape):
            self.variable.data["acceleration"] = self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_units / (self.position_units / self.time_units**2)
        else:
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach() / self.variable.data["mass"][..., None] * self.force_units / (self.position_units / self.time_units**2)
            )
        self.variable.data["velocity"] = self.variable.data["velocity"] + 0.5 * dt * self.variable.data["acceleration"]


class LangevinDynamics(VariableUpdater):
    """
    Implements the Langevin algorithm
    """

    required_variable_data = ["position", "velocity", "mass"]

    def __init__(
        self,
        force_db_name: str,
        temperature_K: float,
        frix: float,
        force_units: Optional[float] = None,
        position_units: Optional[float] = None,
        time_units: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        :param force_db_name: key which will correspond to the force on the corresponding Variable
            in the HIPNN model output dictionary
        :param temperature: temperature for Langevin algorithm
        :param frix: friction coefficient for Langevin algorithm
        :param force_units: model force units output (in terms of ase.units), defaults to eV/Ang
        :param position_units: model position units output (in terms of ase.units), defaults to Ang
        :param time_units: model time units output (in terms of ase.units), defaults to fs
        :param seed: used to set seed for reproducibility, defaults to None

        mass of attached Variable must be in amu
        """

        self.force_key = force_db_name
        self.temperature = temperature_K
        self.frix = frix
        self.force_units = (force_units or ase.units.eV/ase.units.Ang)
        self.position_units = (position_units or ase.units.Ang)
        self.time_units = (time_units or ase.units.fs)
        self.kB = ase.units.kB / (self.force_units * self.position_units)

        if seed is not None:
            torch.manual_seed(seed)

    def pre_step(self, dt:float):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        """

        self.variable.data["position"] = self.variable.data["position"] + self.variable.data["velocity"] * dt

        if "cell" in self.variable.data.keys():
            _, self.variable.data["position"], *_ = wrap_systems_torch(coords=self.variable.data["position"], cell=self.variable.data["cell"], cutoff=0) # cutoff only impacts unused outputs; can be set arbitrarily
            try:
                self.variable.data["unwrapped_position"] = self.variable.data["unwrapped_position"] + self.variable.data["velocity"] * dt
            except KeyError:
                self.variable.data["unwrapped_position"] = self.variable.data["position"].clone().detach()

    def post_step(self, dt: float, model_outputs: dict):
        """
        Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :param model_outputs: dictionary of HIPNN model outputs
        """

        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)

        if len(self.variable.data["force"].shape) != len(self.variable.data["mass"].shape):
            self.variable.data["mass"] = self.variable.data["mass"][..., None]

        self.variable.data["acceleration"] = self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_units / (self.position_units / self.time_units**2)

        frix_term1 = self.frix * self.variable.data["velocity"] * dt

        #              |------units of energy------| |------units of mass-------| |-units cancel-|
        frix_term2 = 2 * self.kB * self.temperature / self.variable.data["mass"] * dt * self.frix
        frix_term2 = frix_term2 * (self.force_units * self.time_units) / (self.position_units / self.time_units) # convert units for energy/mass to units for velocity
        frix_term2 = (frix_term2) ** (1/2)

        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + dt * self.variable.data["acceleration"]
            - frix_term1
            + frix_term2
            * torch.randn_like(self.variable.data["velocity"], memory_format=torch.contiguous_format)
        )

class ASELangevinDynamics(VariableUpdater):
    """
    Implements the Langevin algorithm from the ASE codebase
    """

    required_variable_data = ["position", "velocity", "mass"]

    def __init__(
        self,
        force_db_name: str,
        temperature_K: float,
        frix: float,
        force_units: Optional[float] = None,
        position_units: Optional[float] = None,
        time_units: Optional[float] = None,
        fix_cm: Optional[bool] = True,
        seed: Optional[int] = None,
    ):
        """
        :param force_db_name: key which will correspond to the force on the corresponding Variable
            in the HIPNN model output dictionary
        :param temperature_K: temperature for Langevin algorithm in Kelvin
        :param frix: friction coefficient for Langevin algorithm
        :param force_units: model force units output (in terms of ase.units), defaults to eV/Ang
        :param position_units: model position units output (in terms of ase.units), defaults to Ang
        :param time_units: model time units output (in terms of ase.units), defaults to fs
        :param fix_cm: include adjustment to keep COM fixed, defaults to True
        :param seed: used to set seed for reproducibility, defaults to None
        mass of attached Variable must be in amu
        """

        self.force_key = force_db_name
        self.temperature = temperature_K
        self.frix = frix
        self.force_units = (force_units or ase.units.eV/ase.units.Ang)
        self.position_units = (position_units or ase.units.Ang)
        self.time_units = (time_units or ase.units.fs)
        self.fix_cm = fix_cm
        self.kB = ase.units.kB / (self.force_units * self.position_units)

        if seed is not None:
            torch.manual_seed(seed)


    def pre_step(self, dt:float):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation
        :param dt: timestep
        """

        if len(self.variable.data["velocity"].shape) != len(self.variable.data["mass"].shape):
            self.variable.data["mass"] = self.variable.data["mass"].unsqueeze(-1)

        #          |------units of energy------| |-1/time-| |------units of mass------|
        sigma = 2 * self.temperature * self.kB * self.frix / self.variable.data["mass"]
        sigma = sigma * (self.force_units * self.time_units) / (self.position_units / self.time_units) # convert units for energy/mass to units for velocity
        sigma = (sigma) ** (1/2)
        
        self.c1 = dt / 2 - (dt**2) * self.frix / 8 
        self.c2 = dt * self.frix / 2 - (dt**2) * (self.frix**2) / 8 
        self.c3 = (dt**(1/2)) * sigma / 2 - (dt**(1.5)) * self.frix * sigma / 8 
        self.c5 = (dt**(1.5)) * sigma / (2 * (3**(1/2)))
        self.c4 = self.frix / 2 * self.c5

        xi = torch.randn_like(self.variable.data["velocity"], memory_format=torch.contiguous_format)
        eta = torch.randn_like(self.variable.data["velocity"], memory_format=torch.contiguous_format)
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_cm:
            self.rnd_pos -= self.rnd_pos.mean(axis=1)
            mass = self.variable.data["mass"].clone().detach()
            self.rnd_vel -= (self.rnd_vel * mass).mean(axis=1) / mass

        self.variable.data["velocity"] += self.c1 * self.variable.data["acceleration"] - self.c2 * self.variable.data["velocity"] + self.rnd_vel        
        self.variable.data["position"] = self.variable.data["position"] + self.variable.data["velocity"] * dt + self.rnd_pos

        if "cell" in self.variable.data.keys():
            _, self.variable.data["position"], *_ = wrap_systems_torch(coords=self.variable.data["position"], cell=self.variable.data["cell"], cutoff=0) # cutoff only impacts unused outputs; can be set arbitrarily
            try:
                self.variable.data["unwrapped_position"] = self.variable.data["unwrapped_position"] + self.variable.data["velocity"] * dt
            except KeyError:
                self.variable.data["unwrapped_position"] = self.variable.data["position"].clone().detach()        

    def post_step(self, dt: float, model_outputs: dict):
        """
        Updates to variables performed during each step of MD simulation after HIPNN model evaluation
        :param dt: timestep
        :param model_outputs: dictionary of HIPNN model outputs
        """

        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)

        self.variable.data["acceleration"] = self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_units / (self.position_units / self.time_units**2)

        self.variable.data["velocity"] += self.c1 * self.variable.data["acceleration"] - self.c2 * self.variable.data["velocity"] + self.rnd_vel


class MolecularDynamics:
    """
    Driver for MD run
    """

    def __init__(
        self,
        variables: list[Variable],
        model: Predictor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: Optional[int] = None
    ):
        """
        :param variables: list of Variable objects which will be tracked during simulation
        :param model: HIPNN Predictor
        :param device: device to move variables and model to, defaults to None
        :param dtype: dtype to convert all float type variable data and model parameters to, defaults to None
        :param batch_size: batch size passed to model.__call__
        """

        self.variables = variables
        self.model = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self._data = dict()

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        if not isinstance(variables, list):
            variables = [variables]
        for variable in variables:
            if variable.updater is None:
                raise ValueError(f"Variable with name {variable.name} does not have a VariableUpdater set.")

        variable_names = [variable.name for variable in variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError(f"Duplicate name found for Variables. Each Variable must have a distinct name. Names found: {variable_names}")

        batch_sizes = set([value.shape[0] for variable in variables for value in variable.data.values()])
        if len(batch_sizes) > 1:
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_sizes}. The first axis of each array in 'data' represents a batch axis."
            )

        self._variables = variables

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        input_db_names = [node.db_name for node in model.inputs]
        variable_data_db_names = [key for variable in self.variables for key in variable.model_input_map.keys()]
        for db_name in input_db_names:
            if db_name not in variable_data_db_names:
                raise ValueError(
                    f"Model requires input for '{db_name}', but no Variable found which contains an entry for '{db_name}' in its 'model_input_map'."
                    + f" Entries in the 'model_input_map' should have the form 'hipnn-db_name: variable-data-key' where 'hipnn-db_name'"
                    + f" refers to the db_name of an input for the hippynn Predictor model,"
                    + f" and 'variable-data-key' corresponds to a key in the 'data' dictionary of one of the Variables."
                    + f" Currently assigned db_names are: {variable_data_db_names}."
                )
        self._model = model

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device is None:
            self._device = None
            return
        self._device = device
        self.model.to(device)
        for variable in self.variables:
            variable.to(device)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype is None:
            self._dtype = None
            return
        self._dtype = dtype
        self.model.to(dtype)
        for variable in self.variables:
            variable.to(dtype)

    @singledispatchmethod
    def to(self, arg):
        raise ValueError(f"Argument must be of type torch.device or torch.dtype, but provided argument is of type {type(arg)}.")

    @to.register
    def _(self, arg: torch.device):
        self.device = arg

    @to.register
    def _(self, arg: torch.dtype):
        self.dtype = arg

    def _step(
        self,
        dt: float,
    ):
        for variable in self.variables:
            variable.updater.pre_step(dt)

        model_inputs = {
            hipnn_db_name: variable.data[variable_key]
            for variable in self.variables
            for hipnn_db_name, variable_key in variable.model_input_map.items()
        }
        model_inputs['batch_size'] = self.batch_size

        model_outputs = self.model(**model_inputs)

        for variable in self.variables:
            variable.updater.post_step(dt, model_outputs)

        return model_outputs

    def _update_data(self, model_outputs: dict):
        for variable in self.variables:
            for key, value in variable.data.items():
                try:
                    self._data[f"{variable.name}_{key}"].append(value.cpu().detach())
                except KeyError:
                    self._data[f"{variable.name}_{key}"] = [value.cpu().detach()]
        for key, value in model_outputs.items():
            try:
                self._data[f"output_{key}"].append(value.cpu().detach())
            except KeyError:
                self._data[f"output_{key}"] = [value.cpu().detach()]

    def run(self, dt: float, n_steps: int, record_every: Optional[int] = None):
        """Run `n_steps` of MD algorithm.

        :param dt: timestep
        :param n_steps: number of steps to execute
        :param record_every: frequency at which to store the data at a step in memory,
            record_every = 1 means every step will be stored, defaults to None
        """

        for i in progress_bar(range(n_steps), miniters=np.ceil(n_steps/1000)):
            model_outputs = self._step(dt)
            if record_every is not None and (i + 1) % record_every == 0:
                self._update_data(model_outputs)

    def get_data(self):
        """Returns a dictionary of the recorded data"""
        return {key: torch.stack(value) for key, value in self._data.items()}

    def reset_data(self):
        """Clear all recorded data"""
        self._data = {key: [] for key in self._data.keys()}
