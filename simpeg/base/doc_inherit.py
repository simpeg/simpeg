import inspect
import itertools
import re
import collections
import importlib
import functools
from re import Match
from typing import Callable, Optional, Dict, Tuple, List, Any, Set, Iterator
import warnings

__all__ = ["bind_signature_to_function", "DoceratorMeta", "DocstringInheritWarning"]


class DocstringInheritWarning(ImportWarning):
    pass


REPLACE_REGEX = re.compile(r"%\((?P<replace_key>.*)\)")
REPLACE_STAR_REGEX = re.compile(r"%\((?P<class_name>\S+)\.\*\)")
_numpydoc_sections = [
    # not super interested in these first three sections.
    # "Signature"
    # "Summary"
    # "Extended Summary"
    "Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
    "index",
]
_section_regexs = []
for section in _numpydoc_sections:
    section_regex = rf"(?:(?:^|\n){section}\n-{{{len(section)}}}\n(?P<{section.lower().replace(' ', '_')}>[\s\S]*?))"
    _section_regexs.append(section_regex)

# The numpy regexes require a cleaned docstring
# first gets the contents of each section (assuming they are in order)
NUMPY_SECTION_REGEX = re.compile(
    rf"^(?P<summary>[\s\S]+?)??{'?'.join(_section_regexs)}?$"
)
# Next parses for "arg : type" items.
NUMPY_ARG_TYPE_REGEX = re.compile(
    r"^(?P<arg_name>\S.*?)(?:\s*:\s*(?P<type>.*?))?$", re.MULTILINE
)

ARG_SPLIT_REGEX = re.compile(r"\s*,\s*")


def _pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b, fillvalue=None)


def _parse_numpydoc_parameters(
    doc: str, double_check: bool = True
) -> Iterator[Match[str]]:
    """Parse a numpydoc string for parameter descriptions.

    Parameters
    ----------
    doc : str

    Yields
    ------
    arg : str
        The name of the argument
    type_string: str or None
        arg's type description (if there was one)
    description : str or None
        arg's extended description (if there was one)
    """
    doc = inspect.cleandoc(doc)
    doc_sections = NUMPY_SECTION_REGEX.search(doc).groupdict()
    parameters = doc_sections.get("parameters")
    if parameters is None:
        if double_check and "Parameters\n-" in doc:
            raise TypeError(
                "Unable to parse docstring for parameters section, but it looks like there might be a "
                "'Parameters' section. Did you not put the correct number of `-` on the line below it? "
                "Are the sections in the correct order?",
            )
        parameters = ""

    others = doc_sections.get("other_parameters")
    if double_check and others is None and "Other Parameters\n-" in doc:
        raise TypeError(
            "Unable to parse docstring for other parameters section, but it looks like there "
            "might be an `Other Parameters` section. Did you not put the correct number of `-` on the "
            "line below it? Are the sections in the correct order?",
        )
    if others:
        parameters += "\n" + others

    for match, next_match in _pairwise(NUMPY_ARG_TYPE_REGEX.finditer(parameters)):
        arg, type_string = match.groups()

        # +1 removes the newline character at the end of the argument : type_name match
        start = match.end() + 1
        # -1 removes the newline character at the start of the argument : type_name match (if there was one).
        end = next_match.start() - 1 if next_match is not None else None
        description = parameters[start:end]
        if description == "":
            description = None
        for arg_part in ARG_SPLIT_REGEX.split(arg):
            yield arg_part, type_string, description


def _class_arg_doc_dict(cls: type) -> collections.OrderedDict[str, Dict[str, Any]]:
    parameters = _parse_numpydoc_parameters(cls.__doc__)

    arg_dict = collections.OrderedDict()
    init_params = inspect.signature(cls.__init__).parameters
    for arg, type_string, description in parameters:
        # skip over values that are to be replaced (if there are any left)
        # and skip over *args, **kwargs arguments
        if not REPLACE_REGEX.match(arg) and arg[0] != "*":
            if not (param := init_params.get(arg, None)):
                # This should likely be switched to an error in the future once all classes are properly documented.
                warnings.warn(
                    f"Documented argument {arg}, is not in the signature of {cls.__name__}.__init__",
                    DocstringInheritWarning,
                    stacklevel=2,
                )
            arg_dict[arg] = {
                "type_string": type_string,
                "description": description,
                "parameter": param,
            }
    return arg_dict


def _replace_doc_args(
    doc: str, target: str, args: List[str], infos: List[Dict[str, Any]]
) -> str:

    # must escape all the special regex characters.
    indent = re.search(rf"^\s*(?={re.escape(target)})", doc, re.MULTILINE)[0]

    n_args = len(args)
    n_info = len(infos)
    if n_info == 1:
        type_string = infos[0]["type_string"]
        desc_string = infos[0]["description"]

        # multiple arguments will share the same type string and description
        insert_string = ", ".join(args)
        if type_string:
            insert_string += f" : {type_string}"
        if desc_string:
            insert_string += "\n" + desc_string
    elif n_args == n_info:
        # each argument will get its own type string and description
        insert_pieces = []
        for arg, info in zip(args, infos):
            type_string = info["type_string"]
            desc_string = info["description"]

            insert_piece = arg
            if type_string:
                insert_piece += f" : {type_string}"
            if desc_string:
                insert_piece += "\n" + desc_string
            insert_pieces.append(insert_piece)
        insert_string = "\n".join(insert_pieces)
    else:
        raise ValueError(
            f"Incompatible length of infos: {n_info}. Must be length 1 or have the same length as args: {n_args}"
        )
    # add the indent before the replacement target to every line after the first line
    # in the replacement string.
    replace_string = f"\n{indent}".join(insert_string.splitlines())
    doc = doc.replace(target, replace_string)
    return doc


def _doc_replace(cls: type, star_excludes: Set[str]) -> Tuple[str, inspect.Signature]:
    doc = cls.__doc__
    # replacement items in doc
    args_to_insert = [
        match.group("replace_key") for match in REPLACE_REGEX.finditer(doc)
    ]

    call_parameters = collections.OrderedDict()
    init = cls.__dict__.get("__init__", None)
    if init:
        call_sign = inspect.signature(cls.__init__)
    else:
        # empty call sign
        call_sign = inspect.Signature()

    if not args_to_insert:
        # if nothing to replace... exit early.
        return doc, call_sign

    kwargs_param = None
    for name, parameter in call_sign.parameters.items():
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            call_parameters[name] = parameter
        else:
            kwargs_param = parameter

    super_doc_dict = None
    bases = cls.__mro__[1:]

    for arg_insert in args_to_insert:
        args = []
        replacement = {}
        for item in ARG_SPLIT_REGEX.split(arg_insert):
            class_name, arg = item.rsplit(".", 1)
            # build the super doc dictionary if we will need it.
            # either for specific items, or a * include of all of them.
            if class_name == "super" and not super_doc_dict:
                super_doc_dict = collections.OrderedDict()
                for base in bases[:-1]:  # don't bother checking `object`
                    if base_arg_dict := getattr(base, "_arg_dict", None):
                        super_doc_dict.update(base_arg_dict)
            if arg[0] != "*":
                # do not process *, *args, or **kwargs parameters
                if class_name == "super":
                    arg_dict = super_doc_dict
                    if arg not in super_doc_dict:
                        raise TypeError(
                            f"Argument {arg} not found in {cls.__name__}'s inheritance"
                        )
                else:
                    # import the class and get it's arg_dict
                    try:
                        module_name, m_class_name = class_name.rsplit(".", 1)
                    except ValueError:
                        raise ValueError(
                            f"{class_name} does not include the module information. "
                            f"Should be included as module.to.import.from.{class_name}"
                        )
                    try:
                        target_cls = getattr(
                            importlib.import_module(module_name), m_class_name
                        )
                    except ImportError:
                        raise TypeError(
                            f"Unable to import class {class_name} for docstring replacement"
                        ) from None
                    if target_cls not in bases:
                        raise TypeError(
                            f"{target_cls.__name__} is not a parent of {cls.__name__}"
                        )
                    arg_dict = getattr(target_cls, "_arg_dict", None)
                    if arg_dict is None:
                        raise TypeError(
                            f"{target_cls} must have an _arg_dict attribute"
                        )
                    if arg not in arg_dict:
                        raise TypeError(
                            f"{arg}'s description not found in {target_cls}._arg_dict"
                        )
                replacement = arg_dict[arg]
                args.append(arg)
                if arg not in call_parameters:
                    if not (param := replacement["parameter"]):
                        param = param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
                    else:
                        # create a generic parameter description
                        param = inspect.Parameter(
                            arg, kind=inspect.Parameter.KEYWORD_ONLY, default=None
                        )
                    call_parameters[arg] = param
        if args:
            doc = _replace_doc_args(doc, f"%({arg_insert})", args, [replacement])

    star_args_classes = [
        match.group("class_name") for match in REPLACE_STAR_REGEX.finditer(doc)
    ]

    for star_class in star_args_classes:
        if star_class == "super":
            kwargs_param = None
            star_arg_dict = super_doc_dict
        else:
            module_name, class_name = star_class.rsplit(".", 1)
            try:
                target_cls = getattr(importlib.import_module(module_name), class_name)
            except ImportError:
                raise ImportError(
                    f"Unable to import class {class_name} for docstring replacement"
                )
            if target_cls not in bases:
                raise TypeError(
                    f"{target_cls.__name__} is not a parent of {cls.__name__}"
                )
            star_arg_dict = getattr(target_cls, "_arg_dict", None)
            if not star_arg_dict:
                raise TypeError(f"{target_cls} must have an _arg_dict.")

        replacements = []
        args = []
        for arg, replacement in star_arg_dict.items():
            if arg not in star_excludes and arg not in call_parameters:
                args.append(arg)
                replacements.append(replacement)
                param = replacement["parameter"]
                call_parameters[arg] = param.replace(
                    kind=inspect.Parameter.KEYWORD_ONLY
                )

        doc = _replace_doc_args(doc, f"%({star_class}.*)", args, replacements)

    if kwargs_param:
        call_parameters[kwargs_param.name] = kwargs_param
    new_signature = inspect.Signature(parameters=call_parameters.values())
    return doc, new_signature


def bind_signature_to_function(
    signature: inspect.Signature, func: Callable
) -> Callable:
    """Binds a callable function to a new signature.

    Parameters
    ----------
    signature : inspect.Signature
        The new signature to bind the function to.
    func : callable
        The function to bind the new signature to.

    Returns
    -------
    wrapped : callable
        The wrapped function will raise a `TypeError` if the inputs do not match
        the new signature.
    """

    # Note this function will not raise a `TypeError`, but the function returned
    # from this function will. Thus, `TypeError` is not included in the Raises doc section.
    @functools.wraps(func)
    def bind_signature(*args, **kwargs):
        try:
            params = signature.bind(*args, **kwargs)
        except TypeError as err:
            raise TypeError(f"{func.__qualname__}(): {err}") from None
        return func(*params.args, **params.kwargs)

    bind_signature.__signature__ = signature
    return bind_signature


class DoceratorMeta(type):
    """Metaclass that implements class constructor argument replacement.

    When a target class uses this as a metaclass, it will trigger a replacement on that
    target class's docstring for specific keys. It looks for replacement strings of the form:

    >>> "%(class_name.arg)"

    ``class_name`` can be either:
        1) A specific class from the target class's inheritance tree, in which case it must be in the format:
           ``f"%({class.__module__}.{class.__qualname__})``, or
        2) the special name ``super``, which triggers a lookup for the first instance
           of ``arg`` in the target class's method resolution order.

    ``arg`` can be either:
        1) A specific argument, or
        2) the ``*`` character, which will include everything from ``class_name``, except for
           arguments in ``star_excludes`` or already in the target class's ``__init__`` signature.

    Notes
    -----
    This metaclass assumes that the target class's docstring follows the numpydoc style format.

    Any parameter that is in the target class's __init__ signature will never be pulled in with a `"*"` import.
    It is expected to either be documented on the target class's docstring or explicitly included from a parent.

    Examples
    --------
    We have a simple base class that uses the DoceratorMeta. Its subclasses then have access to the
    argument descriptions in its docstring.

    >>> from simpeg.base import DoceratorMeta
    >>> class BaseClass(metaclass=DoceratorMeta):
    ...     '''A simple base class
    ...
    ...     Parameters
    ...     ----------
    ...     info : str
    ...         Information about this instance.
    ...
    ...     Other Parameters
    ...     ----------------
    ...     more_info : list of str, optional
    ...         Additional information
    ...     '''
    ...     def __init__(self, info, more_info=None):...

    Next we want to creat a new class that inherits from ``BaseClass`` but we don't want to copy and
    paste the description of the `item` argument. We also want to include all of the other arguments
    described in `BaseClass` in this class's Other Parameters section.
    >>> class ChildClass(BaseClass):
    ...     '''A Child Class
    ...     %(super.info)
    ...     %(super.*)
    ...     '''
    ...     def __init__(self, info, **kwargs):...
    >>> print(ChildClass.__doc__)
    A Child Class
    info : str
        Information about this instance.
    more_info : list of str, optional
        Additional information

    You can exclude arguments from wildcard includes (``*``) by setting the `star_excludes` keyword argument
    for that class.
    >>> class OtherChildClass(BaseClass, star_excludes=["more_info"]):
    ...     '''Another child class
    ...     %(super.info)
    ...     %(super.*)
    ...     '''
    ...     def __init__(self, info, **kwargs):...
    >>> print(OtherChildClass.__doc__)
    Another child class
        info : str
            Information about this instance.
    """

    def __new__(
        mcs,
        name,
        bases,
        namespace,
        star_excludes: Optional[set] = None,
        update_signature: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str
        bases : type
        namespace : dict
        star_excludes : set, optional
            Arguments to exclude from any (class_name.*) imports
        update_signature : bool, optional
            Whether to update the class's signature to match the updated docstring.
        **kwargs
            Extra keyword arguments passed to the parent metaclass.
        """
        # construct the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # build the argument dictionary
        if not getattr(cls, "__doc__", None):
            # if I don't have a __doc__ don't do anything.
            cls._arg_dict = {}
            return cls

        cls._arg_dict = _class_arg_doc_dict(cls)
        if star_excludes is None:
            star_excludes = set()
        else:
            # Make a copy, so we don't mutate the input argument.
            star_excludes = set(star_excludes).copy()

        cls._excluded_parent_args = star_excludes
        # get all the excludes from the inheritance tree as well.
        parent_excludes = set()
        for base in cls.__mro__[1:-1]:
            if excluded := getattr(base, "_excluded_parent_args", None):
                parent_excludes.update(excluded)
        excludes = star_excludes | parent_excludes  # set union
        cls.__doc__, init_sig = _doc_replace(cls, excludes)
        # Need to look inside __dict___ to check if the class
        # actually has an __init__ defined for it. Can't check
        # for cls.__init__ because it could pull the parent's
        # __init__ function
        init_func = cls.__dict__.get("__init__", None)
        if init_func and update_signature:
            cls.__init__ = bind_signature_to_function(init_sig, init_func)
        return cls


# Could also add this functionality as a wrapper for a class.
