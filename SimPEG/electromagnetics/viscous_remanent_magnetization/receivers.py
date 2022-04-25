from ...survey import BaseRx
import numpy as np
import warnings
# import properties

#########################################
# POINT RECEIVER CLASS FOR VRM
#########################################


class Point(BaseRx):
    """Point receiver for viscous remanent magnetization simulations

    Parameters
    ----------
    locations : (n, 3) numpy.ndarray
        Receiver locations
    times : numpy.ndarray
        Time channels
    field_type : str
        Fields being measured. Choose from {'h', 'b', 'dhdt', 'dbdt'
    orientation : str, default = 'z'
        Receiver orientation. Choose from {'x', 'y', 'z'}
    """

    # times = properties.Array("Observation times", dtype=float)

    # fieldType = properties.StringChoice(
    #     "Field type", choices=["h", "b", "dhdt", "dbdt"]
    # )

    # orientation = properties.StringChoice(
    #     "Component of response", choices=["x", "y", "z"]
    # )

    def __init__(self, locations=None, times=None, field_type=None, orientation='z', **kwargs):

        super(Point, self).__init__(locations=locations, **kwargs)

        fieldType = kwargs.pop("fieldType", None)
        if fieldType is not None:
            warnings.warn(
                "'fieldType' is a deprecated property. Please use 'field_type' instead."
                "'fieldType' be removed in SimPEG 0.17.0."
            )
            field_type = fieldType
        if field_type is None:
            raise AttributeError("VRM receiver class cannot be instantiated witout 'field_type")
        else:
            self.field_type = field_type
        
        if times is None:
            raise AttributeError("VRM receiver class cannot be instantiated without 'times'")
        else:
            self.times = times

        self.orientation = orientation

    @property
    def times(self):
        """Time channels for the receivers

        Returns
        -------
        numpy.ndarray
            Time channels for the receivers
        """
        return self._times

    @times.setter
    def times(self, value):
        # Ensure float or numpy array of float
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"times is not a valid type. Got {type(value)}")
        
        if value.ndim > 1:
            raise TypeError(f"times must be ('*') array")

        self._times = value

    @property
    def orientation(self):
        """Orientation of the receiver.

        Returns
        -------
        str
            Orientation of the receiver. One of {'x', 'y', 'z'}
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):

        if isinstance(var, str):
            var = var.lower()
            if var not in ('x', 'y', 'z'):
                raise ValueError(f"orientation must be either 'x', 'y' or 'z'. Got {var}")
        else:
            raise TypeError(f"orientation must be a str. Got {type(var)}")

        self._orientation = var

    @property
    def field_type(self):
        """Field type; i.e. 'h', 'b', 'dhdt' or 'dbdt'

        Returns
        -------
        str
            Field type; i.e. 'h', 'b', 'dhdt' or 'dbdt'
        """
        return self._field_type

    @field_type.setter
    def field_type(self, var):
        if isinstance(var, str):
            var = var.lower()
            if var not in ('h', 'b', 'dhdt', 'dbdt'):
                raise ValueError(f"field_type must be either 'h', 'b', 'dhdt', 'dbdt'. Got {var}")
        else:
            raise TypeError(f"field_type must be a str. Got {type(var)}")

        self._field_type = var

    @property
    def n_times(self):
        """Number of time channels

        Returns
        -------
        int
            Number of time channels
        """
        if self.times is not None:
            return len(self.times)

    @property
    def n_locations(self):
        """Number of receivers

        Returns
        -------
        int
            Number of receivers
        """
        return self.locations.shape[0]

    @property
    def nD(self):
        """Number of data associated with the receiver

        Returns
        -------
        int
            Number of data associated with the receiver
        """
        if self.times is not None:
            return self.locations.shape[0] * len(self.times)


class SquareLoop(Point):
    """Square loop receiver

    Measurements with this type of receiver are the field, integrated over the
    area of the loop, then multiplied by the number of coils, then normalized
    by the dipole moment. As a result, the units for fields predicted with this
    type of receiver are the same as 'h', 'b', 'dhdt' and 'dbdt', respectively.
    
    Parameters
    ----------
    locations : (n, 3) numpy.ndarray
        Center location of the square loop
    times : numpy.ndarray
        Time channels
    field_type : str
        Fields being measured. Choose from {'h', 'b', 'dhdt', 'dbdt'
    orientation : str, default = 'z'
        Receiver orientation. Choose from {'x', 'y', 'z'}
    width : float, default = 1.0
        Loop width (m)
    n_turns : int, default = 1
        Number of turns for the receiver coil
    quadrature_order : int, default = 3
        Order of numerical quadrature for approximating the magnetic flux through
        the receiver coil.
    
    """

    # width = properties.Float("Square loop width", min=1e-6)
    # nTurns = properties.Integer("Number of loop turns", min=1, default=1)
    # quadOrder = properties.Integer(
    #     "Order for numerical quadrature integration over loop", min=1, max=7, default=3
    # )

    def __init__(
        self,
        locations=None,
        times=None,
        field_type=None,
        orientation='z',
        width=1.0,
        n_turns=1,
        quadrature_order=3,
        **kwargs):

        if 'nTurns' in kwargs:
            warnings.warn(
                "'nTurns' is a deprecated property. Please use 'n_turns' instead."
                "'nTurns' be removed in SimPEG 0.17.0."
            )
            n_turns = kwargs.pop('nTurns')

        if 'quadOrder' in kwargs:
            warnings.warn(
                "'quadOrder' is a deprecated property. Please use 'quadrature_order' instead."
                "'quadOrder' be removed in SimPEG 0.17.0."
            )
            quadrature_order = kwargs.pop('quadOrder')

        super(SquareLoop, self).__init__(
            locations=locations, times=times, field_type=field_type, orientation=orientation, **kwargs
        )

        self.width = width
        self.n_turns = n_turns
        self.quadrature_order = quadrature_order

    @property
    def width(self):
        """Square loop width

        Returns
        -------
        float
            Square loop width
        """
        return self._width

    @width.setter
    def width(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(
                f"width must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        if value <= 0.:
            raise ValueError("Width must be positive")
        self._width = value

    @property
    def n_turns(self):
        """Number of turns in the receiver coil

        Returns
        -------
        int
            Number of turns in the receiver coil
        """
        return self._n_turns

    @n_turns.setter
    def n_turns(self, value):
        if not isinstance(value, int):
            TypeError(
                f"n_turns must be int, the value provided, {value} is "
                f"{type(value)}"
            )

        if value < 1:
            raise ValueError("'n_turns' must be a positive integer")
        else:
            self._n_turns = value

    @property
    def quadrature_order(self):
        """Quadrature order for determining flux through receiver coil

        Returns
        -------
        int
            Quadrature order for determining flux through receiver coil
        """
        return self._quadrature_order

    @quadrature_order.setter
    def quadrature_order(self, value):
        if not isinstance(value, int):
            TypeError(
                f"quadrature_order must be int, the value provided, {value} is "
                f"{type(value)}"
            )

        if value < 1:
            raise ValueError("'quadrature_order' must be a positive integer")
        elif value > 7:
            raise ValueError("Largest 'quadrature_order' implemented is 7")
        else:
            self._quadrature_order = value       
