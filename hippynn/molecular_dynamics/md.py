from __future__ import annotations
from functools import singledispatchmethod
from copy import copy

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
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        """
        :param name: name for variable
        :type name: str
        :param data: dictionary of tracked data in the form `value_name: value`
        :type data: dict[str, torch.Tensor]
        :param model_input_map: dictionary of correspondences between data tracked by Variable
            and inputs to the HIP-NN model in the form
            `hipnn-db_name: variable-data-key`, defaults to dict()
        :type model_input_map: dict[str, str], optional
        :param updater: object which will update the data of the Variable
            over the course of the MD simulation, defaults to None
        :type updater: VariableUpdater, optional
        :param device: device on which to keep data, defaults to None
        :type device: torch.device, optional
        :param dtype: dtype for float type data, defaults to None
        :type dtype: torch.dtype, optional
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

    def pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        """        
        pass

    def post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        :param model_outputs: dictionary of HIPNN model outputs
        :type model_outputs: dict
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
        units_force: float = ase.units.eV,
        units_acc: float = ase.units.Ang / (1.0**2),
    ):
        """
        :param force_db_name: key which will correspond to the force on the corresponding Variable
            in the HIPNN model output dictionary
        :type force_db_name: str
        :param units_force: amount of eV equal to one in the units used for force output
            of HIPNN model (eg. if force output in kcal, units_force =
            ase.units.kcal = 2.6114e22 since 2.6114e22 kcal = 1 eV),
            by default ase.units.eV = 1, defaults to ase.units.eV
        :type units_force: float, optional
        :param units_acc: amount of Ang/fs^2 equal to one in the units used for acceleration
            in the corresponding Variable, by default units.Ang/(1.0 ** 2) = 1, defaults to ase.units.Ang/(1.0**2)
        :type units_acc: float, optional
        """        
        self.force_key = force_db_name
        self.force_factor = units_force / units_acc

    def pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        """        
        self.variable.data["velocity"] = self.variable.data["velocity"] + 0.5 * dt * self.variable.data["acceleration"]
        self.variable.data["position"] = self.variable.data["position"] + self.variable.data["velocity"] * dt
        try:
            _, self.variable.data["position"], *_ = wrap_systems_torch(coords=self.variable.data["position"], cell=self.variable.data["cell"], cutoff=0) # cutoff only used for discarded outputs; can be set arbitrarily
        except KeyError:
            pass

    def post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        :param model_outputs: dictionary of HIPNN model outputs
        :type model_outputs: dict
        """        
        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)
        if len(self.variable.data["force"].shape) == len(self.variable.data["mass"].shape):
            self.variable.data["acceleration"] = self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_factor
        else:
            self.variable.data["acceleration"] = (
                self.variable.data["force"].detach() / self.variable.data["mass"][..., None] * self.force_factor
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
        temperature: float,
        frix: float,
        units_force=ase.units.eV,
        units_acc=ase.units.Ang / (1.0**2),
        seed: int = None,
    ):
        """
        :param force_db_name: key which will correspond to the force on the corresponding Variable
            in the HIPNN model output dictionary
        :type force_db_name: str
        :param temperature: temperature for Langevin algorithm
        :type temperature: float
        :param frix: friction coefficient for Langevin algorithm
        :type frix: float
        :param units_force: amount of eV equal to one in the units used for force output
            of HIPNN model (eg. if force output in kcal, units_force =
            ase.units.kcal = 2.6114e22 since 2.6114e22 kcal = 1 eV),
            by default ase.units.eV = 1, defaults to ase.units.eV
        :type units_force: float, optional
        :param units_acc: amount of Ang/fs^2 equal to one in the units used for acceleration
            in the corresponding Variable, by default units.Ang/(1.0 ** 2) = 1, defaults to ase.units.Ang/(1.0**2)
        :type units_acc: float, optional
        :param seed: used to set seed for reproducibility, defaults to None
        :type seed: int, optional
        """        

        self.force_key = force_db_name
        self.force_factor = units_force / units_acc
        self.temperature = temperature
        self.frix = frix
        self.kB = 0.001987204 * self.force_factor

        if seed is not None:
            torch.manual_seed(seed)

    def pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation before HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        """        

        self.variable.data["position"] = self.variable.data["position"] + self.variable.data["velocity"] * dt

        if "cell" in self.variable.data.keys():
            _, self.variable.data["position"], *_ = wrap_systems_torch(coords=self.variable.data["position"], cell=self.variable.data["cell"], cutoff=0) # cutoff only impacts unused outputs; can be set arbitrarily
            try:
                self.variable.data["unwrapped_position"] = self.variable.data["unwrapped_position"] + self.variable.data["velocity"] * dt
            except KeyError:
                self.variable.data["unwrapped_position"] = copy(self.variable.data["position"])
    def post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation after HIPNN model evaluation

        :param dt: timestep
        :type dt: float
        :param model_outputs: dictionary of HIPNN model outputs
        :type model_outputs: dict
        """        

        self.variable.data["force"] = model_outputs[self.force_key].to(self.variable.device)

        if len(self.variable.data["force"].shape) != len(self.variable.data["mass"].shape):
            self.variable.data["mass"] = self.variable.data["mass"][..., None]

        self.variable.data["acceleration"] = self.variable.data["force"].detach() / self.variable.data["mass"] * self.force_factor

        self.variable.data["velocity"] = (
            self.variable.data["velocity"]
            + dt * self.variable.data["acceleration"]
            - self.frix * self.variable.data["velocity"] * dt
            + torch.sqrt(2 * self.kB * self.frix * self.temperature / self.variable.data["mass"] * dt)
            * torch.randn_like(self.variable.data["velocity"], memory_format=torch.contiguous_format)
        )


class MolecularDynamics:
    """
    Driver for MD run
    """

    def __init__(
        self,
        variables: list[Variable],
        model: Predictor,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        :param variables: list of Variable objects which will be tracked during simulation
        :type variables: list[Variable]
        :param model: HIPNN Predictor
        :type model: Predictor
        :param device: device to move variables and model to, defaults to None
        :type device: torch.device, optional
        :param dtype: dtype to convert all float type variable data and model parameters to, defaults to None
        :type dtype: torch.dtype, optional
        """        

        self.variables = variables
        self.model = model
        self.device = device
        self.dtype = dtype

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

        model_outputs = self.model(**model_inputs)

        for variable in self.variables:
            variable.updater.post_step(dt, model_outputs)

        return model_outputs

    def _update_data(self, model_outputs: dict):
        for variable in self.variables:
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
        

    def run(self, dt: float, n_steps: int, record_every: int = None):
        """Run `n_steps` of MD algorithm.

        :param dt: timestep
        :type dt: float
        :param n_steps: number of steps to execute
        :type n_steps: int
        :param record_every: frequency at which to store the data at a step in memory,
            record_every = 1 means every step will be stored, defaults to None
        :type record_every: int, optional
        """        

        for i in progress_bar(range(n_steps)):
            model_outputs = self._step(dt)
            if record_every is not None and (i + 1) % record_every == 0:
                self._update_data(model_outputs)

    def get_data(self):
        """Returns a dictionary of the recorded data"""
        return {key: torch.stack(value) for key, value in self._data.items()}

    def reset_data(self):
        """Clear all recorded data"""
        self._data = {key: [] for key in self._data.keys()}
