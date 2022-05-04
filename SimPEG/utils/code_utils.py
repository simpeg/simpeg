from __future__ import print_function, division
import types
import numpy as np
from functools import wraps
import warnings
import properties

from discretize.utils import asArray_N_x_Dim

# scooby is a soft dependency for SimPEG
try:
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:
        def __init__(self, additional, core, optional, ncol, text_width, sort):
            print(
                "\n  *ERROR*: `SimPEG.Report` requires `scooby`."
                "\n           Install it via `pip install scooby` or"
                "\n           `conda install -c conda-forge scooby`.\n"
            )


def create_wrapper_from_class(input_class, *fun_names):
    """Create wrapper class with memory profiler.

    Using :meth:`memory_profiler.profile`, this function creates a wrapper class
    from the input class and function names specified.
    
    Parameters
    ----------
    input_class : class
        Input class being used to create the wrapper
    fun_names : list of str
        Names of the functions that will be wrapped to the wrapper class. These names must
        correspond to methods of the input class.  
    
    Returns
    -------
    class :
        Wrapper class

    Examples
    --------

    >>> foo_mem = create_wrapper_from_class(foo,['my_func'])
    >>> fooi = foo_mem()
    >>> for i in range(5):
    >>>     fooi.my_func()

    Then run it from the command line

    ``python -m memory_profiler exampleMemWrapper.py``
    """
    from memory_profiler import profile

    attrs = {}
    for f in fun_names:
        if hasattr(input_class, f):
            attrs[f] = profile(getattr(input_class, f))
        else:
            print("{0!s} not found in {1!s} Class".format(f, input_class.__name__))

    return type(input_class.__name__ + "MemProfileWrap", (input_class,), attrs)


def hook(obj, method, name=None, overwrite=False, silent=False):
    """Dynamically bind a class's method to an instance of a different class.

    Parameters
    ----------
    obj : class
        Instance of a class that will be binded to a new method
    method : method
        The method that will be binded to *obj*. The syntax is *ClassName.method*
    name : str, optional
        Provide a different name for the method being binded to *obj*. If ``None``,
        the original method name is used. 
    overwrite : bool, default: ``False``
        If ``True``, the hook will overwrite a preexisting method of *obj* if it has
        the same name as the *name* input argument. If ``False``, preexisting methods
        are not overwritten.
    silent : bool, default: ``False``
        Print whether a previous hook was overwritten
    """
    if name is None:
        name = method.__name__
        if name == "<lambda>":
            raise Exception("Must provide name to hook lambda functions.")
    if not hasattr(obj, name) or overwrite:
        setattr(obj, name, types.MethodType(method, obj))
        if getattr(obj, "debug", False):
            print("Method " + name + " was added to class.")
    elif not silent or getattr(obj, "debug", False):
        print("Method " + name + " was not overwritten.")


def set_kwargs(obj, ignore=None, **kwargs):
    """
    Set key word arguments for an object or throw an error if any do not exist.
    
    Parameters
    ----------
    obj : class
        Instance of a class
    ignore : list, optional
        ``list`` of ``str`` denoting kwargs that are ignored (not being set)
    """
    if ignore is None:
        ignore = []
    for attr in kwargs:
        if attr in ignore:
            continue
        if hasattr(obj, attr):
            setattr(obj, attr, kwargs[attr])
        else:
            raise Exception("{0!s} attr is not recognized".format(attr))

    # hook(obj, hook, silent=True)
    # hook(obj, setKwargs, silent=True)


def print_done(obj, printers, name="Done", pad=""):
    """Print completion of an operation (**DOCSTRING INCOMPLETE**)

    Parameters
    ----------
    obj : object
        An object
    printers : list of dict
        Has keys "width" and "title"
    name : str, default: "Done"
        A string for the process being completed
    pad : str, default: ""
        Trailing string

    """
    titles = ""
    widths = 0
    for printer in printers:
        titles += ("{{:^{0:d}}}".format(printer["width"])).format(printer["title"]) + ""
        widths += printer["width"]
    print(pad + "{0} {1} {0}".format("=" * ((widths - 1 - len(name)) // 2), name))
    # print(pad + "%s" % '-'*widths)


def print_titles(obj, printers, name="Print Titles", pad=""):
    """Print titles (**DOCSTRING INCOMPLETE**)

    Parameters
    ----------
    obj : object
        An object
    printers : list of dict
        Has keys "width" and "title"
    name : str, default: "Print Titles"
        A string for the process being completed
    pad : str, default: ""
        Trailing string
    """
    titles = ""
    widths = 0
    for printer in printers:
        titles += ("{{:^{0:d}}}".format(printer["width"])).format(printer["title"]) + ""
        widths += printer["width"]
    print(pad + "{0} {1} {0}".format("=" * ((widths - 1 - len(name)) // 2), name))
    print(pad + titles)
    print(pad + "%s" % "-" * widths)


def print_line(obj, printers, pad=""):
    """Print line (**DOCSTRING INCOMPLETE**)

    Parameters
    ----------
    obj : object
        An object
    printers : list of dict
        Has keys "width" and "title"
    pad : str, default: ""
        Trailing string
    """
    values = ""
    for printer in printers:
        values += ("{{:^{0:d}}}".format(printer["width"])).format(
            printer["format"] % printer["value"](obj)
        )
    print(pad + values)


def check_stoppers(obj, stoppers):
    """Check stopping rules (**DOCSTRING INCOMPLETE**)
    
    Parameters
    ----------
    obj : object
        Input object
    stoppers : list of dict
        List of stoppers

    Returns
    -------
    bool :
        Whether stopping criteria was encountered
    """
    optimal = []
    critical = []
    for stopper in stoppers:
        l = stopper["left"](obj)
        r = stopper["right"](obj)
        if stopper["stopType"] == "optimal":
            optimal.append(l <= r)
        if stopper["stopType"] == "critical":
            critical.append(l <= r)

    if obj.debug:
        print("checkStoppers.optimal: ", optimal)
    if obj.debug:
        print("checkStoppers.critical: ", critical)

    return (len(optimal) > 0 and all(optimal)) | (len(critical) > 0 and any(critical))


def print_stoppers(obj, stoppers, pad="", stop="STOP!", done="DONE!"):
    """Print stoppers (**DOCSTRING INCOMPLETE**)

    Parameters
    ----------
    obj : object
        An object
    stoppers : list of dict
        Has keys "width" and "title"
    pad : str, default: ""
        Trailing string
    stop : str, default: "STOP!"
        String for statement when stopping criteria encountered
    done : str, default: "DONE!"
        String for statement when stopping criterian not encountered
    """
    print(pad + "{0!s}{1!s}{2!s}".format("-" * 25, stop, "-" * 25))
    for stopper in stoppers:
        l = stopper["left"](obj)
        r = stopper["right"](obj)
        print(pad + stopper["str"] % (l <= r, l, r))
    print(pad + "{0!s}{1!s}{2!s}".format("-" * 25, done, "-" * 25))


def call_hooks(match, mainFirst=False):
    """Wrap a function to an instance of a class (**DOCSTRING INCOMPLETE**)

    Use the following syntax::

        @callHooks('doEndIteration')
        def doEndIteration(self):
            pass

    This will call everything named *_doEndIteration* at the beginning of the function call.
    By default the main method (doEndIteration) is run after all of the sub methods (_doEndIteration*).
    This can be reversed by adding the mainFirst=True kwarg.

    Parameters
    ----------
    match : str
        Name of the function being wrapped to class instance
    mainFirst : bool, default: ``False``
        Main first

    Returns
    -------
    wrapper
        The wrapper
    """

    def callHooksWrap(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):

            if not mainFirst:
                for method in [
                    posible for posible in dir(self) if ("_" + match) in posible
                ]:
                    if getattr(self, "debug", False):
                        print((match + " is calling self." + method))
                    getattr(self, method)(*args, **kwargs)

                return f(self, *args, **kwargs)
            else:
                out = f(self, *args, **kwargs)

                for method in [
                    posible for posible in dir(self) if ("_" + match) in posible
                ]:
                    if getattr(self, "debug", False):
                        print((match + " is calling self." + method))
                    getattr(self, method)(*args, **kwargs)

                return out

        extra = """
            If you have things that also need to run in the method {0!s}, you can create a method::

                def _{1!s}*(self, ... ):
                    pass

            Where the * can be any string. If present, _{2!s}* will be called at the start of the default {3!s} call.
            You may also completely overwrite this function.
        """.format(
            match, match, match, match
        )
        doc = wrapper.__doc__
        wrapper.__doc__ = ("" if doc is None else doc) + extra
        return wrapper

    return callHooksWrap


def dependent_property(name, value, children, doc):
    """Dependent property (**DOCSTRING INCOMPLETE**)

    Parameters
    ----------
    name : str
        Property name
    value : scalar
        A scalar value
    children : class instances
        Child classes
    doc : str
        Property documentation
    """
    def fget(self):
        return getattr(self, name, value)

    def fset(self, val):
        if (np.isscalar(val) and getattr(self, name, value) == val) or val is getattr(
            self, name, value
        ):
            return  # it is the same!
        for child in children:
            if hasattr(self, child):
                delattr(self, child)
        setattr(self, name, val)

    return property(fget=fget, fset=fset, doc=doc)


def requires(var):
    """Wrap a function (**DOCSTRING INCOMPLETE**)

    Use the following syntax to wrap a funciton::

        @requires('prob')
        def dpred(self):
            pass

    This wrapper will ensure that a problem has been bound to the data.
    If a problem is not bound an Exception will be raised, and an nice error message printed.

    Parameters
    ----------
    var :
        Input variable

    Returns
    -------
    wrapper
        The wrapper
    """

    def requiresVar(f):
        if var == "prob":
            extra = """

        .. note::

            To use survey.{0!s}(), SimPEG requires that a problem be bound to the survey.
            If a problem has not been bound, an Exception will be raised.
            To bind a problem to the Data object::

                survey.pair(myProblem)

            """.format(
                f.__name__
            )
        else:
            extra = """
                To use *{0!s}* method, SimPEG requires that the {1!s} be specified.
            """.format(
                f.__name__, var
            )

        @wraps(f)
        def requiresVarWrapper(self, *args, **kwargs):
            if getattr(self, var, None) is None:
                raise Exception(extra)
            return f(self, *args, **kwargs)

        doc = requiresVarWrapper.__doc__
        requiresVarWrapper.__doc__ = ("" if doc is None else doc) + extra

        return requiresVarWrapper

    return requiresVar


class Report(ScoobyReport):
    """Print date, time, and version information.

    Use scooby to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``SimPEG``, ``cython``, ``properties``, ``vectormath``, ``discretize``,
    ``pymatsolver``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``,
    ``matplotlib``, and ``ipywidgets``. It also shows MKL information, if
    available.

    All modules provided in ``add_pckg`` are also shown.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).
    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.
    text_width : int, optional
        The text width for non-HTML display modes
    sort : bool, optional
        Sort the packages when the report is shown

    Examples
    --------

    >>> import pytest
    >>> import dateutil
    >>> from SimPEG import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Define nr of columns

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = [
            "SimPEG",
            "discretize",
            "pymatsolver",
            "vectormath",
            "properties",
            "numpy",
            "scipy",
            "cython",
        ]

        # Optional packages.
        optional = ["IPython", "matplotlib", "ipywidgets"]

        super().__init__(
            additional=add_pckg,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )


##############################################################
#               DEPRECATION FUNCTIONS
##############################################################

def deprecate_class(
    removal_version=None, new_location=None, future_warn=False, error=False
):
    """Utility function to deprecate a class

    Parameters
    ----------
    removal_version : str
        A string denoting the SimPEG version in which the class will be removed
    new_location : str
        Name for the class replacing the deprecated class
    future_warn : bool, default: ``False``
        If ``True``, throw comprehensive warning the class will be deprecated
    error : bool, default: ``False``
        Throw error if deprecated class no longer implemented

    Returns
    -------
    class
        The new class 
    """
    def decorator(cls):
        my_name = cls.__name__
        parent_name = cls.__bases__[0].__name__
        message = f"{my_name} has been deprecated, please use {parent_name}."
        if error:
            message = f"{my_name} has been removed, please use {parent_name}."
        elif removal_version is not None:
            message += f" It will be removed in version {removal_version} of SimPEG."
        else:
            message += " It will be removed in a future version of SimPEG."

        # stash the original initialization of the class
        cls._old__init__ = cls.__init__

        def __init__(self, *args, **kwargs):
            if future_warn:
                warnings.warn(message, FutureWarning)
            elif error:
                raise NotImplementedError(message)
            else:
                warnings.warn(message, DeprecationWarning)
            self._old__init__(*args, **kwargs)

        cls.__init__ = __init__
        if new_location is not None:
            parent_name = f"{new_location}.{parent_name}"
        cls.__doc__ = f""" This class has been deprecated, see `{parent_name}` for documentation"""
        return cls

    return decorator


def deprecate_module(
    old_name, new_name, removal_version=None, future_warn=False, error=False
):
    """Deprecate module

    Parameters
    ----------
    old_name : str
        Original name for the now deprecated module
    new_name : str
        New name for the module
    removal_version : str, optional
        SimPEG version in which the module will be removed from the code base
    future_warn : bool, default: ``False``
        If ``True``, throw comprehensive warning the module will be deprecated
    error : bool, default: ``False``
        Throw error if deprecated module no longer implemented
    """
    message = f"The {old_name} module has been deprecated, please use {new_name}."
    if error:
        message = f"{old_name} has been removed, please use {new_name}."
    elif removal_version is not None:
        message += f" It will be removed in version {removal_version} of SimPEG"
    else:
        message += " It will be removed in a future version of SimPEG."
    message += " Please update your code accordingly."
    if future_warn:
        warnings.warn(message, FutureWarning)
    elif error:
        raise NotImplementedError(message)
    else:
        warnings.warn(message, DeprecationWarning)


def deprecate_property(
    prop, old_name, new_name=None, removal_version=None, future_warn=False, error=False
):
    """Deprecate property

    Parameters
    ----------
    prop : property
        Current property
    old_name : str
        Original name for the now deprecated property
    new_name : str, optional
        New name for the property. If ``None``, the property name is take from the
        *prop* input argument.
    removal_version : str, optional
        SimPEG version in which the property will be removed from the code base
    future_warn : bool, default: ``False``
        If ``True``, throw comprehensive warning the property will be deprecated
    error : bool, default: ``False``
        Throw error if deprecated property no longer implemented

    Returns
    -------
    property
        The new property
    """

    if isinstance(prop, property):
        if new_name is None:
            new_name = prop.fget.__qualname__
        cls_name = new_name.split(".")[0]
        old_name = f"{cls_name}.{old_name}"
    elif isinstance(prop, properties.GettableProperty):
        if new_name is None:
            new_name = prop.name
        prop = prop.get_property()

    message = f"{old_name} has been deprecated, please use {new_name}."
    if error:
        message = f"{old_name} has been removed, please use {new_name}."
    elif removal_version is not None:
        message += f" It will be removed in version {removal_version} of SimPEG."
    else:
        message += " It will be removed in a future version of SimPEG."

    def get_dep(self):
        if future_warn:
            warnings.warn(message, FutureWarning)
        elif error:
            raise NotImplementedError(message)
        else:
            warnings.warn(message, DeprecationWarning)
        return prop.fget(self)

    def set_dep(self, other):
        if future_warn:
            warnings.warn(message, FutureWarning)
        elif error:
            raise NotImplementedError(message)
        else:
            warnings.warn(message, DeprecationWarning)
        prop.fset(self, other)

    doc = f"`{old_name}` has been deprecated. See `{new_name}` for documentation"

    return property(get_dep, set_dep, prop.fdel, doc)


def deprecate_method(
    method, old_name, removal_version=None, future_warn=False, error=False
):
    """Deprecate method

    Parameters
    ----------
    method : method
        Current method
    old_name : str
        Original name for the now deprecated method
    removal_version : str, optional
        SimPEG version in which the method will be removed from the code base
    future_warn : bool, default: ``False``
        If ``True``, throw comprehensive warning the method will be deprecated
    error : bool, default: ``False``
        Throw error if deprecated method no longer implemented

    Returns
    -------
    method
        The new method
    """
    new_name = method.__qualname__
    split_name = new_name.split(".")
    if len(split_name) > 1:
        old_name = f"{split_name[0]}.{old_name}"

    message = f"{old_name} has been deprecated, please use {new_name}."
    if error:
        message = f"{old_name} has been removed, please use {new_name}."
    elif removal_version is not None:
        message += f" It will be removed in version {removal_version} of SimPEG."
    else:
        message += " It will be removed in a future version of SimPEG."

    def new_method(*args, **kwargs):
        if future_warn:
            warnings.warn(message, FutureWarning)
        elif error:
            raise NotImplementedError(message)
        else:
            warnings.warn(message, DeprecationWarning)
        return method(*args, **kwargs)

    doc = f"`{old_name}` has been deprecated. See `{new_name}` for documentation"
    new_method.__doc__ = doc
    return new_method


def deprecate_function(new_function, old_name, removal_version=None):
    """Deprecate function

    Parameters
    ----------
    new_function : function
        Current function
    old_name : str
        Original name for the now deprecated function
    removal_version : str, optional
        SimPEG version in which the method will be removed from the code base
    future_warn : bool, default: ``False``
        If ``True``, throw comprehensive warning the method will be deprecated
    error : bool, default: ``False``
        Throw error if deprecated method no longer implemented

    Returns
    -------
    function
        The new function
    """
    new_name = new_function.__name__
    if removal_version is not None:
        tag = f" It will be removed in version {removal_version} of SimPEG."
    else:
        tag = " It will be removed in a future version of SimPEG."

    def dep_function(*args, **kwargs):
        warnings.warn(
            f"{old_name} has been deprecated, please use {new_name}." + tag,
            DeprecationWarning,
        )
        return new_function(*args, **kwargs)

    doc = f"""
    `{old_name}` has been deprecated. See `{new_name}` for documentation

    See Also
    --------
    {new_name}
    """
    dep_function.__doc__ = doc
    return dep_function


###############################################################
#                    PROPERTY VALIDATORS
###############################################################

def validate_string_property(property_name, var, string_list=None):
    """Validate a string property

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : str
        The input variable
    string_list : list or tuple of str, optional
        Provide a list of acceptable strings

    Returns
    -------
    str
        Returns the input argument *var* once validated 
    """

    if isinstance(var, str):
        if string_list is None:
            return var
        else:
            if var in string_list:
                return var
            else:
                raise ValueError(f"'{property_name}' must in '{string_list}'. Got '{var}'")
    else:
        raise TypeError(f"'{property_name}' must be a str. Got '{type(var)}'")
    

def validate_integer_property(property_name, var, min_val=-np.inf, max_val=np.inf):
    """Validate integer property

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : int or float
        The input variable
    min_val : int or float, optional
        Minimum value
    max_val : int or float, optional
        Maximum value

    Returns
    -------
    float
        Returns the input variable as a float once validated
    """
    try:
        var = int(var)
    except:
        raise TypeError(f"'{property_name}' must be int or float, got '{type(var)}'")

    if (var < min_val) | (var > max_val):
        raise ValueError(f"'{property_name}' must be a value between {min_val} and {max_val}")
    else:
        return var

def validate_float_property(property_name, var, min_val=-np.inf, max_val=np.inf):
    """Validate float property

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : int or float
        The input variable
    min_val : int or float, optional
        Minimum value
    max_val : int or float, optional
        Maximum value

    Returns
    -------
    float
        Returns the input variable as a float once validated
    """
    try:
        var = float(var)
    except:
        raise TypeError(f"'{property_name}' must be int or float, got '{type(var)}'")

    if (var < min_val) | (var > max_val):
        raise ValueError(f"'{property_name}' must be a value between {min_val} and {max_val}")
    else:
        return var


def validate_list_property(property_name, var, class_type):
    """Validate list of instances of a certain class

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : object or a list of object
        A list of objects
    class_type : class or tuple of class types
        Class type(s) that are allowed in the list

    Returns
    -------
    list
        Returns the list once validated
    """
    if isinstance(var, class_type):
        var = [var]
    elif isinstance(var, list):
        pass
    else:
        raise TypeError(f"'{property_name}' must be a list of '{class_type}'")

    is_true = [isinstance(x, class_type) for x in var]
    if np.all(is_true):
        return var
    else:
        TypeError(f"'{property_name}' must be a list of '{class_type}'")


def validate_location_property(property_name, var, dim=None):
    """Validate a location

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : numpy.array_like
        The input variable
    dim : int, optional
        The dimension; i.e. 1, 2 or 3

    Returns
    -------
    numpy.ndarray
        Returns the location once validated
    """
    try:
        var = np.atleast_1d(var).astype(float).squeeze()
    except:
        raise TypeError(f"'{property_name}' must be array_like, got {type(var)}")

    if dim is None:
        return var
    else:
        if len(var) == dim:
            return var
        else:
            raise ValueError(
                f"'{property_name}' must be array_like with shape '{dim}', got '{len(var)}'"
            )


def validate_ndarray_property(property_name, var, shape=('*', '*'), dtype=float):
    """Validate numerical array property

    Parameters
    ----------
    property_name : str
        The name of the property being set
    var : numpy.ndarray
        The input array
    shape : tuple of int, default: ('*', '*')
        The shape of the array; e.g. (3), (3, 3), ('*', 2).
        The '*' indicates that an arbitrary number of elements is allowed
        along a particular dimension.
    dtype : float (default), int, complex, bool
        The data type for the array

    Returns
    -------
    numpy.ndarray
        Returns the array in the specified data type once validated
    """
    try:
        if len(shape) == 1:
            var = np.atleast_1d(var).astype(dtype)
        elif len(shape) == 2:
            var = np.atleast_2d(var).astype(dtype)
        elif len(shape) == 3:
            var = np.atleast_3d(var).astype(dtype)
        else:
            raise NotImplementedError("Only implemented for 1D, 2D and 3D arrays!!!")
    except:
        raise TypeError(f"'{property_name}' must be {shape} array_like, got {type(var)}")

    for ii, value in np.shape(var):
        if (shape[ii] != '*') & (shape[ii] != value):
            raise ValueError(f"'{property_name}' must be {shape}, got {np.shape(var)}")

    return var


###############################################################
#                      DEPRECATIONS
###############################################################
memProfileWrapper = deprecate_function(create_wrapper_from_class, "memProfileWrapper", removal_version="0.16.0")
setKwargs = deprecate_function(set_kwargs, "setKwargs", removal_version="0.16.0")
printTitles = deprecate_function(print_titles, "printTitles", removal_version="0.16.0")
printLine = deprecate_function(print_line, "printLine", removal_version="0.16.0")
printStoppers = deprecate_function(print_stoppers, "printStoppers", removal_version="0.16.0")
checkStoppers = deprecate_function(check_stoppers, "checkStoppers", removal_version="0.16.0")
printDone = deprecate_function(print_done, "printDone", removal_version="0.16.0")
callHooks = deprecate_function(call_hooks, "callHooks", removal_version="0.16.0")
dependentProperty = deprecate_function(dependent_property, "dependentProperty", removal_version="0.16.0")
