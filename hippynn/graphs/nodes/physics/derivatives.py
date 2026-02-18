import warnings


from ....layers import physics as physics_layers
from ...indextypes import IdxType 
from ..base import (
    AutoKw,
    AutoNoKw,
    ExpandParents,
    MultiNode,
    SingleNode,
    Node,
    find_unique_relative,
)
from ..indexers import  acquire_encoding_padding
from ..inputs import PositionsNode
from ..tags import Encoder, Energies


class GradientNode(AutoKw, SingleNode):
    """
    Compute the gradient of a quantity.
    """

    input_names = "energy", "coordinates"
    auto_module_class = physics_layers.Gradient
    auto_module_kwargs = "sign",

    def __init__(self, name, parents, sign, **kwargs):
        energy, position = parents
        position.requires_grad = True
        parents = energy.main_output, position
        self.sign = sign
        self.index_state = position.index_state
        super().__init__(name, parents, sign=sign, **kwargs)

        
class MultiGradientNode(AutoKw, MultiNode):
    """
    Compute the gradient of a quantity.
    """

    auto_module_class = physics_layers.MultiGradient
    auto_module_kwargs = "signs",


    def __init__(self, name: str, molecular_energies_parent: Node, generalized_coordinates_parents: tuple[Node], signs: tuple[int], **kwargs):
        if isinstance(signs, int):
            signs = (signs,)

        self.signs = signs

        parents = molecular_energies_parent, *generalized_coordinates_parents

        for parent in generalized_coordinates_parents:
            parent.requires_grad = True        

        self.input_names = tuple((parent.name for parent in parents))
        self.output_names = tuple((parent.name + "_grad" for parent in generalized_coordinates_parents))
        self.output_index_states = tuple(parent.index_state for parent in generalized_coordinates_parents)

        super().__init__(name, parents, signs=signs, **kwargs)


class HessianNode(ExpandParents, AutoKw, MultiNode):
    """
    Node that computes the Hessian (second derivatives of energy)
    via gradients of force w.r.t. coordinates or
    second gradients of enery w.r.t. coordinates.
    """

    input_names = "forces", "coordinates", "nonblank"
    output_names = "hessian", "mask"
    output_index_states = (IdxType.Molecules, IdxType.Molecules)
    auto_module_class = physics_layers.Hessian

    @parent_expander.matchlen(1)
    def expansion0(self, source, *, purpose, **kwargs):
        # Infer positions from energy or force node
        return source, find_unique_relative(source, PositionsNode, why_desc=purpose)

    @parent_expander.match(Energies, PositionsNode)
    def expansion1(self, energy, positions, *, purpose, **kwargs):
        energy = energy.main_output
        possible_grads = [child for child in energy.children if (isinstance(child, GradientNode) and child.coordinates == positions)]

        if len(possible_grads) == 1:
            # if we found a unique gradient, use that
            force = possible_grads[0]
        elif len(possible_grads)==0:
            # if no gradient was found, make our own
            force = GradientNode("forces", (energy, positions), sign=-1)
        elif len(possible_grads)>1:
            raise NodeAmbiguityError("Unable to automatically determine correct gradient of energy, as multiple gradient nodes are already present.")

        return force, positions

    @parent_expander.match(GradientNode, PositionsNode)
    def expansion2(self, force, coordinates, *, purpose, **kwargs):
        # always use forces, not gradients
        if force.sign == +1:
            force = -1 * force
        return force, coordinates

    @parent_expander.match(Node, PositionsNode)
    def expansion3(self, force, coordinates, *, purpose, **kwargs):
        encoder, _ = acquire_encoding_padding((force, coordinates), species_set=None, purpose=purpose)
        return force, coordinates, encoder

    @parent_expander.match(Node, PositionsNode, Encoder)
    def expansion4(self, force, coordinates, encoder, **kwargs):
        coordinates.requires_grad = True
        return force, coordinates, encoder.nonblank

    @parent_expander.match(Node, Node, Node)
    def check_for_grad(self, force, coordinates, encoder, **kwargs):
        self_and_parents = {force, *force.get_ancestors()}
        if not any(isinstance(n, (GradientNode, MultiGradientNode)) for n in self_and_parents):
            warnings.warn(f"Input to hessian node doesn't appear to be a force or child of a force! This node: {force}")
        return force, coordinates, encoder

    parent_expander.assertlen(3)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.MolAtom, IdxType.MolAtom, None)


    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        self._index_state = IdxType.Molecules
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class HVPVectorNode(ExpandParents, AutoKw, SingleNode):
    """
    Outputs a tensor with a determined number of random or one-hot vectors per molecule
    """
    input_names = "coordinates", "nonblank"
    index_state = IdxType.MolAtom
    auto_module_class = physics_layers.HVPVector

    @parent_expander.match(PositionsNode)
    def expand_from_positions(self, positions, *, purpose=None, **kwargs):
        encoder, _ = acquire_encoding_padding((positions,), species_set=None, purpose=purpose)
        return positions, encoder.nonblank

    parent_expander.assertlen(2)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.MolAtom, IdxType.MolAtom)

    def __init__(self, name, parents,  module="auto", vector_type="random", **kwargs):
        self.module_kwargs = {"vector_type": vector_type}
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class HVPNode(ExpandParents, AutoKw, MultiNode):
    input_names = "source", "coordinates", "vector", "nonblank"
    output_names = "hvp", "mask"
    output_index_states = (IdxType.MolAtom, IdxType.MolAtom)
    auto_module_class = physics_layers.HVP

    @parent_expander.match(Energies, Node)
    def expansion0(self, source, vector, *, purpose, **kwargs):
        # Infer positions from energy or force node
        positions = find_unique_relative(source, PositionsNode, why_desc=purpose)
        return source, positions, vector

    @parent_expander.match(Energies, PositionsNode, Node)
    def expansion1(self, energy, positions, vector, *, purpose, **kwargs):
        energy = energy.main_output
        possible_grads = [child for child in energy.children if (isinstance(child, GradientNode) and child.coordinates == positions)]

        if len(possible_grads) == 1:
            # if we found a unique gradient, use that
            force = possible_grads[0]
        elif len(possible_grads)==0:
            # if no gradient was found, make our own
            force = GradientNode("forces", (energy, positions), sign=-1)
        elif len(possible_grads)>1:
            raise NodeAmbiguityError("Unable to automatically determine gradient of energy as multiple gradient nodes are present.")

        return force, positions, vector

    @parent_expander.match(GradientNode, PositionsNode, Node)
    def expansion2(self, force, positions, vector, *, purpose, **kwargs):
        # always use forces, not gradients
        if force.sign == +1:
            force = -1 * force
        return force, positions, vector

    @parent_expander.match(Node, PositionsNode, Node)
    def expansion3(self, force, positions, vector, *, purpose, **kwargs):

        if not isinstance(force, GradientNode) and not any(isinstance(f, GradientNode) for f in force.get_all_parents()):
            warnings.warn(f"Input to HVP node doesn't appear to be a force or child of a force! Got node: {force}")

        encoder, _ = acquire_encoding_padding((force, positions), species_set=None, purpose=purpose)
        return force, positions, vector, encoder

    @parent_expander.match(Node, PositionsNode, Node, Encoder)
    def expansion4(self, force, coordinates, vector, encoder, *, purpose, **kwargs):
        coordinates.requires_grad = True
        return force, coordinates, vector, encoder.nonblank

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.MolAtom, IdxType.MolAtom, IdxType.MolAtom, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        self._index_state = IdxType.Molecules
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class TrueHVPNode(ExpandParents, AutoNoKw, SingleNode):
    """
    Computes true Hessian-vector product from database-stored Hessians and input vector
    """
    input_names = "hessian", "vector"
    index_state = IdxType.MolAtom
    auto_module_class = physics_layers.TrueHVP

    @parent_expander.match(Node, HVPVectorNode)
    def expand_from_hessian_and_vector(self, hessian, vector, **kwargs):
        if hessian._index_state != IdxType.Molecules:
            raise TypeError(f"Expected Molecules-indexed Hessian, got {hessian._index_state}")
        return hessian, vector

    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Molecules, IdxType.MolAtom)

    def __init__(self, name, parents=None, module="auto", **kwargs):
        self.module_kwargs = {}
        parents = self.expand_parents(parents, **kwargs)
        super().__init__(name, parents, module=module, **kwargs)


class StressForceNode(AutoNoKw, MultiNode):
    input_names = "energy", "strain", "coordinates", "cell"
    output_names = "forces", "stress"
    auto_module_class = physics_layers.StressForce

    def __init__(self, name, parents, module="auto", **kwargs):
        energy, strain, coordinates, cell = parents
        coordinates.requires_grad = True
        parents = energy.main_output, strain, coordinates, cell
        self.output_index_states = coordinates.index_state, strain.index_state
        super().__init__(name, parents, module=module, **kwargs)

    
class StrainInducer(AutoNoKw, MultiNode):
    input_names = "coordinates", "cell"
    output_names = "strained_coordinates", "strained_cell", "strain"
    output_index_states = NotImplemented
    auto_module_class = physics_layers.CellScaleInducer

    def __init__(self, name, parents, module="auto", **kwargs):
        position, cell = parents
        self.output_index_states = position.index_state, IdxType.Unlabeled, IdxType.Unlabeled
        super().__init__(name, parents, module=module, **kwargs)

    
def setup_stressforce_nodes(energy_node, return_transformed_inputs=False, positions_node="auto", cell_node="auto", strain_node="auto"):
    """_summary_

    :param energy_node: the energy to differenitate
    :param return_transformed_inputs: If true, return the strained positions, strained cell, and strain
    :param position_node: defaults to "auto"
    :param cell_node: defaults to "auto"
    :param strain_node: defaults to "auto"

    Using "auto" will cause a failure if the corresponding node cannot be found or is ambiguous.

    :return: (forces, stress) or (forces, stress, strained_positions, strained_cell, strain) depending on return_transformed_inputs flag.
    """


    from ..tags import Positions
    from ..inputs import CellNode
    
    if positions_node == "auto":
        positions_node = find_unique_relative(energy_node, Positions)

    if cell_node == "auto":
        cell_node = find_unique_relative(energy_node, CellNode)
    
    if strain_node == "auto":
        strain_node = StrainInducer("Strain_inducer", (positions_node, cell_node))
    
    strained_coords = strain_node.strained_coordinates
    strained_cell = strain_node.strained_cell
    strain = strain_node.strain

    from hippynn.graphs.gops import replace_node

    replace_node(positions_node, strained_coords)
    replace_node(cell_node, strained_cell)

    derivatives = StressForceNode("StressForceCalculator", (energy_node, strain, positions_node, cell_node))
    forces, stress = derivatives.forces, derivatives.stress

    if return_transformed_inputs:

        return stress, forces, strained_coords, strained_cell, strain
    
    else:
        return stress, forces
