import torch
import time
import copy
from hippynn.experiment import serialization

time_stamps = [time.time()]

# callback function for update scf eps along epochs
def update_scf_eps(seqm, decay_factor=0.98, minimal=27.2114e-6):
    def func(epoch, better_model):
        if better_model:
            print("from callback update_scf_eps")
            p0 = seqm.energy.hamiltonian.eps.detach().data.item()
            p1 = p0 * decay_factor
            if p1 < minimal:
                p1 = minimal
            if p1 < p0:
                dtype = seqm.energy.hamiltonian.eps.dtype
                device = seqm.energy.hamiltonian.eps.device
                seqm.energy.hamiltonian.eps.data = torch.tensor(p1, dtype=dtype, device=device)
                print("SCF eps is updated: ", p0, "==>", p1)

    return func


def update_scf_backward_eps(seqm, decay_factor=0.96, minimal=1e-3):
    def func(epoch, better_model):
        if better_model:
            print("from callback update_scf_backward_eps")
            p0 = seqm.energy.hamiltonian.scf_backward_eps.detach().data.item()
            p1 = p0 * decay_factor
            if p1 < minimal:
                p1 = minimal
            if p1 < p0:
                dtype = seqm.energy.hamiltonian.scf_backward_eps.dtype
                device = seqm.energy.hamiltonian.scf_backward_eps.device
                seqm.energy.hamiltonian.scf_backward_eps.data = torch.tensor(p1, dtype=dtype, device=device)
                print("SCF scf_backward eps is updated: ", p0, "==>", p1)

    return func


def save_and_stop_after(
    training_modules, controller, metric_tracker, store_all_better=False, store_best=True, queue_time=[0, 0, 1, 0]
):
    def func(epoch, better_model):
        time_stamps.append(time.time())
        n_epochs = len(time_stamps) - 1
        time_epoch = [time_stamps[i + 1] - time_stamps[i] for i in range(n_epochs)]
        max_time_per_epoch = max(time_epoch)
        total_time = queue_time[0] * 24.0 * 3600.0 + queue_time[1] * 3600.0 + queue_time[2] * 60.0 + queue_time[3] * 1.0
        available_time = total_time - (time_stamps[-1] - time_stamps[0])
        if available_time < max_time_per_epoch * 2.0:
            # save before stop
            model = training_modules[0]
            with open("last_model.pt", "wb") as pfile:
                torch.save(copy.deepcopy(model.state_dict()), pfile)

            state = serialization.create_state(model, controller, metric_tracker)

            # Write the checkpoint
            with open("last_checkpoint.pt", "wb") as pfile:
                torch.save(state, pfile)

            if better_model:
                print("**** NEW BEST MODEL - Saving! ****")
                best_model = copy.deepcopy(model.state_dict())
                metric_tracker.best_model = best_model

                if store_all_better:
                    # Save a copy of every network doing better
                    # Note: epoch has already been incremented, so decrement in saving file.
                    with open(f"better_model_epoch_{epoch}.pt", "wb") as pfile:
                        torch.save(best_model, pfile)

                if store_best:
                    # Overwrite the "best model so far"
                    with open("best_model.pt", "wb") as pfile:
                        torch.save(best_model, pfile)

                    state = serialization.create_state(model, controller, metric_tracker)

                    # Write the checkpoint
                    with open("best_checkpoint.pt", "wb") as pfile:
                        torch.save(state, pfile)

            print("time for each epoch: ", time_epoch)
            print("Time remaining in sec: ", available_time)
            raise KeyboardInterrupt("time to stop")

    return func
