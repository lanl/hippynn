import warnings

from hippynn.custom_kernels.autograd_wrapper import wrap_envops


class MessagePassingKernels:
    _registered_implementations = {}  # Registry for custom kernel implementations.
    def __new__(cls, impl_name: str, envsum_impl, sensesum_impl, featsum_impl, wrap=True,
                 compiler=None):
        """
        __new__ is specialized so handle the case where compilation errors.
        """
        if compiler is not None:
            try:
                envsum_impl, sensesum_impl, featsum_impl = \
                    map(compiler, (envsum_impl, sensesum_impl, featsum_impl))
            except Exception as e:    
                w = RuntimeWarning(*e.args)
                w.with_traceback(e.__traceback__)
                warnings.warn(w)
                warnings.warn(f"Compilation errored, custom kernel implementation '{impl_name}' will not be available.")
                return None
        return super().__new__(cls)



    def __init__(self, impl_name: str, envsum_impl, sensesum_impl, featsum_impl, wrap=True,
                 compiler=None,):
        """
        :param impl_name: name for implementation.
        :param envsum_impl: non-autograd-wrapped envsum implementation
        :param sensesum_impl: non-autograd-wrapped sensesum implementation
        :param featsum_impl: non-autograd-wrapped featsum implementation
        :param wrap: set to false if implementations are already autograd-capable.
        """

        
        self.envsum_impl = envsum_impl
        self.sensesum_impl = sensesum_impl
        self.featsum_impl = featsum_impl

        if wrap:
            envsum, sensesum, featsum = wrap_envops(envsum_impl, sensesum_impl, featsum_impl)
        else:
            envsum, sensesum, featsum = envsum_impl, sensesum_impl, featsum_impl

        self.envsum = envsum
        self.sensesum = sensesum
        self.featsum = featsum

        impl_name = impl_name.lower()
        if impl_name in self._registered_implementations:
            raise ValueError(f"Already have implementation of kernels named {impl_name}!")
        else:
            self._registered_implementations[impl_name] = self

    @classmethod
    def get_implementation(cls, impl_name):
        """

        :param impl_name:
        :return:
        :raises CustomKernelError if implementation is not available or known.
        """
        from . import CustomKernelError
        try:
            impl = cls._registered_implementations[impl_name.lower()]
        except KeyError:
            raise CustomKernelError(f"Unavailable custom kernel implementation: {impl_name}")
        return impl

    @classmethod
    def get_available_implementations(self, hidden=False):
        """
        Return the available implementations of the custom kernels.

        :param hidden: Show all implementations, even those which have no improved performance characteristics.
        :return:
        """
        
        return [k for k in self._registered_implementations.keys() if hidden or not k.startswith("_")]
