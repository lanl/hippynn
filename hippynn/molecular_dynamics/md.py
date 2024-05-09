from __future__ import annotations
from typing import Union

import numpy as np
import torch

from tqdm.autonotebook import trange
import ase

from ..graphs import GraphModule


class Variable:
    """
    Tracks the state of a quantity (eg. position, cell, species,
    volume) on each particle or each system in an MD simulation. Can
    also hold additional values associated to that quantity (such as
    velocity, acceleration, etc...)
    """

    def __init__(
        self,
        name: str,
        values: dict[str, torch.Tensor],
        model_input_map: dict[str, torch.Tensor] = dict(),
        device: Union[str, torch.device] = None,
    ) -> None:
        """
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

    def set_values(self, values: dict[str, torch.Tensor]):
        if not isinstance(values, dict):
            raise TypeError("The argument for 'values' must be a dictionary.")

        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            if value.shape[0] != 1:  # GraphModule requires inputs in batches
                value.unsqueeze_(0)
            values[key] = value

        try:
            self.values.update(values)
        except AttributeError:
            self.values = values

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Moves values stored in internal variable to 'device'.
        """
        self.device = device
        for key, value in self.values.items():
            self.values[key] = value.to(device)


class StaticVariable(Variable):
    """
    A Variable child class for Variables whose values do not change
    during the MD simulation.
    """

    pass


class DynamicVariable(Variable):
    """
    A Variable child class for Variables whose values change
    during the MD simulation.
    """

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
        name : str
            name for variable
        starting_values : dict[str, torch.Tensor]
            dictionary of tracked values in the form `value_name: value`
        model_input_map : dict[str, str], optional
            dictionary of correspondences between values tracked by Variable
            and values input to HIP-NN model in the form
            `hipnn-model-input-key: variable-value-key`, by default dict()
        updater : DynamicVariableUpdater, optional
            object which will update the values of the Variable
            over the course of the MD simulation, by default None
        device : Union[str, torch.device], optional
            device on which to keep values, by default None
        """
        self.name = name
        self.model_input_map = model_input_map

        self.set_updater(updater)
        self.set_values(starting_values)

        if device is not None:
            self.to(device)

    def set_updater(self, updater: DynamicVariableUpdater):
        self.updater = updater
        if updater is not None:
            updater._attach(self)


class DynamicVariableUpdater:
    """
    Parent class for algorithms to make updates to the values of a Variable during
    each step on an MD simulation.

    Subclasses should redefine __init__, _pre_step, _post_step, and
    required_variable_values as needed. The inputs to _pre_step and _post_step
    should not be changed.
    """

    required_variable_values = (
        []
    )  # A list of keys which must appear in the Variable.values
    # any Variable that will be updated by objects of this class.
    # Checked for in _attach function.

    def __init__(self):
        pass

    def _attach(self, variable: DynamicVariable):
        self.variable = variable
        for value in self.required_variable_values:
            if value not in self.variable.values.keys():
                raise ValueError(
                    f"Cannot attach to Variable with no assigned values for {value}. Use Variable.set_values to add values for {value}."
                )

    def _pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation
          before HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        """
        pass

    def _post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation
          after HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        model_outputs : dict
            dictionary of HIPNN model outputs
        """
        pass


class VelocityVerlet(DynamicVariableUpdater):
    """
    Implements the Velocity Verlet algorithm
    """

    required_variable_values = ["position", "velocity", "acceleration", "mass"]

    def __init__(
        self,
        force_key: str,
        units_force: float = ase.units.eV,
        units_acc: float = ase.units.Ang / (1.0**2),
    ):
        """
        Parameters
        ----------
        force_key : str
            key which will correspond to the force on the modified Variable
            in the HIPNN model output dictionary
        units_force : float, optional
            amount of eV equal to one in the units used for force output
            of HIPNN model (eg. if force output in kcal, units_force =
            ase.units.kcal = 2.6114e22 since 2.6114e22 kcal = 1 eV),
            by default ase.units.eV = 1
        units_acc : float, optional
            amount of Ang/fs^2 equal to one in the units used for acceleration
            in the corresponding Variable, by default units.Ang/(1.0 ** 2) = 1
        """
        self.force_key = force_key
        self.force_factor = units_force / units_acc

    def _pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation
          before HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        """
        self.variable.values["velocity"] = (
            self.variable.values["velocity"]
            + 0.5 * dt * self.variable.values["acceleration"]
        )
        self.variable.values["position"] = (
            self.variable.values["position"] + self.variable.values["velocity"] * dt
        )

    def _post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation
          after HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        model_outputs : dict
            dictionary of HIPNN model outputs
        """
        self.variable.values["force"] = model_outputs[self.force_key].to(
            self.variable.device
        )
        if len(self.variable.values["force"].shape) == len(
            self.variable.values["mass"].shape
        ):
            self.variable.values["acceleration"] = (
                self.variable.values["force"].detach()
                / self.variable.values["mass"]
                * self.force_factor
            )
        else:
            self.variable.values["acceleration"] = (
                self.variable.values["force"].detach()
                / self.variable.values["mass"][..., None]
                * self.force_factor
            )
        self.variable.values["velocity"] = (
            self.variable.values["velocity"]
            + 0.5 * dt * self.variable.values["acceleration"]
        )


class LangevinDynamics(DynamicVariableUpdater):
    """
    Implements the Langevin algorithm
    """

    required_variable_values = ["position", "velocity", "mass"]

    def __init__(
        self,
        force_key: str,
        temperature: float,
        frix: float,
        units_force=ase.units.eV,
        units_acc=ase.units.Ang / (1.0**2),
        seed: int = None,
    ):
        """
        Parameters
        ----------
        force_key : str
            key which will correspond to the force on the modified Variable
            in the HIPNN model output dictionary
        temperature : float
            temperature for Langevin algorithm
        frix : float
            friction coefficient for Langevin algorithm
        units_force : float, optional
            amount of eV equal to one in the units used for force output
            of HIPNN model (eg. if force output in kcal, units_force =
            ase.units.kcal = 2.6114e22 since 2.6114e22 kcal = 1 eV),
            by default ase.units.eV = 1
        units_acc : float, optional
            amount of Ang/fs^2 equal to one in the units used for acceleration
            in the corresponding Variable, by default units.Ang/(1.0 ** 2) = 1
        seed : int, optional
            used to set seed for reproducibility, by default None
        """

        self.force_key = force_key
        self.force_factor = units_force / units_acc
        self.temperature = temperature
        self.frix = frix
        self.kB = 0.001987204 * self.force_factor

        if seed is not None:
            torch.manual_seed(seed)

    def _pre_step(self, dt):
        """Updates to variables performed during each step of MD simulation
          before HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        """

        self.variable.values["position"] = (
            self.variable.values["position"] + self.variable.values["velocity"] * dt
        )

    def _post_step(self, dt, model_outputs):
        """Updates to variables performed during each step of MD simulation
          after HIPNN model evaluation

        Parameters
        ----------
        dt : float
            timestep
        model_outputs : dict
            dictionary of HIPNN model outputs
        """
        self.variable.values["force"] = model_outputs[self.force_key].to(
            self.variable.device
        )

        if len(self.variable.values["force"].shape) != len(
            self.variable.values["mass"].shape
        ):
            self.variable.values["mass"] = self.variable.values["mass"][..., None]

        self.variable.values["acceleration"] = (
            self.variable.values["force"].detach()
            / self.variable.values["mass"]
            * self.force_factor
        )

        self.variable.values["velocity"] = (
            self.variable.values["velocity"]
            + dt * self.variable.values["acceleration"]
            - self.frix * self.variable.values["velocity"] * dt
            + torch.sqrt(
                2
                * self.kB
                * self.frix
                * self.temperature
                / self.variable.values["mass"]
                * dt
            )
            * torch.randn_like(self.variable.values["velocity"])
        )


class MolecularDynamics:
    """
    Driver for MD run
    """

    def __init__(
        self,
        dynamic_variables: list[DynamicVariable],
        static_variables: list[StaticVariable],
        model: GraphModule,
    ):
        """
        Parameters
        ----------
        dynamic_variables : list[DynamicVariable]
            list of DynamicVariable objects which will be tracked during simulation
        static_variables : list[StaticVariable]
            list of StaticVariable objects which will be tracked during simulation
        model : GraphModule
            HIPNN GraphModule (eg. hippynn.graphs.predictor.Predictor object)
        """
        self.dynamic_variables = dynamic_variables
        self.static_variables = static_variables
        self.model = model

        self._data = dict()

    def _step(
        self,
        dt: float,
    ):
        for variable in self.dynamic_variables:
            variable.updater._pre_step(dt)

        model_outputs = self.model(
            **{
                hippynn_key: variable.values[variable_key]
                for variable in [*self.dynamic_variables, *self.static_variables]
                for hippynn_key, variable_key in variable.model_input_map.items()
            }
        )

        for variable in self.dynamic_variables:
            variable.updater._post_step(dt, model_outputs)

        return model_outputs

    def _update_values(self, model_outputs: dict):

        for variable in [*self.dynamic_variables, *self.static_variables]:
            for key, value in variable.values.items():
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
        """
        Run `n_steps` of MD algorithm.

        Parameters
        ----------
        dt : float
            timestep
        n_steps : int
            number of steps to execute
        record_every : int, optional
            frequency at which to store the values at a step in memory,
            record_every = 1 means every step will be stored, by default None
        """
        for i in trange(n_steps):
            model_outputs = self._step(dt)
            if record_every is not None and (i + 1) % record_every == 0:
                self._update_values(model_outputs)

    def get_data(self):
        """Returns a dictionary of the recorded data"""
        return {key: np.array(value) for key, value in self._data.items()}

    def reset_data(self):
        """Clear all recorded data"""
        self._data = {key: [] for key in self._data.keys()}
