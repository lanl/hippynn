"""
Classes for allowing reload of database.

This isn't meant to cover all possible use cases,
but the conveniently allow restarting from a checkpoint
file without manually re-loading the database.

Restarting a database should only be performed
if no preprocessing is applied, i.e. in order
for this functionality to work automatically, you should
save a copy of your database completely preprocessed.
Otherwise you'll need to reproduce the processing
that was applied between constructing the database object
and constructing the checkpoint.
"""


class Restarter:
    def attempt_restart(self):
        return NotImplemented


class NoRestart(Restarter):
    def attempt_restart(self):
        print("Couldn't reload database. It might have been generated in-memory.")
        return None


class RestartDB(Restarter):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        state = self.__dict__.copy()
        cls_spec = (self.cls.__module__, self.cls.__qualname__)
        state["cls"] = cls_spec
        return state

    def __setstate__(self, state):
        cls_state = state["cls"]
        if isinstance(cls_state, tuple):
            cls_module, cls_name = cls_state
            try:
                import importlib

                module = importlib.import_module(cls_module)
                state["cls"] = getattr(module, cls_name)
            except (ImportError, AttributeError) as ee:
                # Save the message, not the full exception object
                # including traceback etc.
                state["cls"] = ee.msg
        for k, v in state.items():
            setattr(self, k, v)

    def attempt_restart(self):
        print("restarting", self.cls)
        if isinstance(self.cls, str):
            raise RuntimeError(f"Not restartable due to class error: {self.cls}")
        try:
            db = self.cls(*self.args, **self.kwargs)
        except Exception as eee:
            raise RuntimeError("Exception while attempting to reload the database.") from eee
        return db


class Restartable:
    @classmethod
    def make_restarter(cls, *args, **kwargs):
        return RestartDB(cls, *args, **kwargs)
