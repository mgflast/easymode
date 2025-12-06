from typing import Union, Tuple, List, Callable, Any
import numpy as np

def sample_scalar(value: Union[Tuple, List, Callable, Any], *args: Any) -> Any:
    """
    Implementation from the batchgenerators package.

    Function to sample a scalar from a specified range, or compute it using
    a provided function.

    Args:
        value: The value to be sampled. It can be a tuple/list defining a range,
               a callable function, or any other value. If it's a tuple/list,
               a random value within the range is returned. If it's a function, it is
               called with the arguments supplied in *args. If it's any other value,
               it is returned as is.
        *args: Additional arguments to be passed to the callable 'value', if it is a
                function.

    Returns
    -------
        A sampled or computed scalar value.
    """
    if isinstance(value, (tuple, list)):
        return np.random.uniform(value[0], value[1])
    elif callable(value):
        # return value(image, kernel)
        return value(*args)
    else:
        return value