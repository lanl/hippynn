import torch
from hippynn.graphs import loss
from hippynn.experiment.serialization import load_checkpoint
from hippynn.experiment.assembly import assemble_for_training
from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode
from .seqm_modules import pack_par

# hippynn.custom_kernels.set_custom_kernels (False)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class gen_par(torch.nn.Module):
    def __init__(
        self,
        model_file="experiment_structure.pt",
        state_file="best_checkpoint.pt",
        par_atom_node_name="SEQM_Atom_Params",
        seqm_node_name="SEQM_Energy",
        device=device,
        weights_only=False,
    ):
        """ 
        .. Warning::
            This function uses ``torch.load`` with ``weights_only=False`` by default,
            which can run arbitrary code from the loaded file. Only use it with files
            you trust. You can set ``weights_only=True`` for better security, but
            it may cause loading of the structure file to fail. 
        """
        super().__init__()
        self.model_file = model_file
        self.state_file = state_file
        self.par_atom_node_name = par_atom_node_name
        self.seqm_node_name = seqm_node_name
        structure = torch.load(model_file, map_location=device, weights_only=weights_only)
        state = torch.load(state_file, map_location=device)
        structure["training_modules"][0].load_state_dict(state["model"])
        structure["controller"].load_state_dict(state["controller"])
        self.model = structure["training_modules"][0]
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.model.to(device)
        self.model.eval()
        par_atom = self.model.node_from_name(par_atom_node_name)
        seqm_par = par_atom.atom_charges
        train_loss = loss.MeanSq(seqm_par.pred)
        validation_losses = {"par_atom": train_loss}
        self.model_par = assemble_for_training(train_loss, validation_losses)[0][0]
        self.model_par.eval()
        if isinstance(self.model_par.input_nodes[0], SpeciesNode):
            self.input_order = True
        else:
            self.input_order = False
        self.seqm_module = self.model.node_from_name(seqm_node_name).torch_module

    def forward(self, species, positions):
        if self.input_order:
            input = [species, positions]
        else:
            input = [positions, species]
        par = self.model_par(*input)[0]
        learned_parameters = pack_par(self.seqm_module, species, par)
        return learned_parameters
