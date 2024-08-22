import inspect
import itertools
import re
import collections
import importlib
import functools
from re import Match
from typing import Callable, Optional, Dict, Tuple, List, Any, Set, Iterator

__all__ = ["bind_signature_to_function", "DoceratorMeta"]

REPLACE_REGEX = re.compile(r"%\((?P<replace_key>.*)\)")
REPLACE_STAR_REGEX = re.compile(r"%\((?P<class_name>\S+)\.\*\)")
_numpydoc_sections = [
    "Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "See Also",
    "Notes",
    "References",
    "Examples",
]
_section_regexs = []
for section in _numpydoc_sections:
    section_regex = rf"(?:(?:^|\n\n){section}\n-{{{len(section)}}}\n(?P<{section.lower().replace(' ', '_')}>[\s\S]*?))"
    _section_regexs.append(section_regex)

NUMPY_SECTION_REGEX = re.compile(
    rf"^(?P<summary>[\s\S]+?)??{'?'.join(_section_regexs)}?$"
)
ARG_TYPE_SEP_REGEX = re.compile(r"\s*:\s*")
ARG_SPLIT_REGEX = re.compile(r"\s*,\s*")
NUMPY_ARG_TYPE_REGEX = re.compile(
    r"^(?P<arg_name>\S.*?)(?:\s*:\s*(?P<type>.*?))?$", re.MULTILINE
)


def _pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b, fillvalue=None)


def _parse_numpydoc_parameters(doc: str) -> Iterator[Match[str]]:
    """Parse a numpydoc string for parameter descriptions.

    Parameters
    ----------
    doc : str

    Yields
    -------
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
        parameters = ""

    others = doc_sections.get("other_parameters")
    if others is not None:
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
            arg_dict[arg] = {
                "type_string": type_string,
                "description": description,
                "parameter": init_params.get(arg, None),
            }
    return arg_dict


def _replace_doc_args(
    doc: str, target: str, args: List[str], infos: List[Dict[str, Any]]
) -> str:

    # must escape all the special regex characters.
    replace_target_regex = re.escape(target)
    indent_search_regex = rf"^\s*(?={replace_target_regex})"
    indent = re.search(indent_search_regex, doc, re.MULTILINE)[0]

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
    doc = re.sub(replace_target_regex, replace_string, doc)
    return doc


def _doc_replace(cls: type, star_excludes: Set[str]) -> Tuple[str, inspect.Signature]:
    doc = cls.__doc__
    # replacement items in doc
    args_to_insert = [
        match.group("replace_key") for match in REPLACE_REGEX.finditer(doc)
    ]

    call_parameters = collections.OrderedDict()
    init = getattr(cls, "__init__", None)
    if init:
        call_sign = inspect.signature(cls.__init__)
    else:
        # empty call sign
        call_sign = inspect.Signature()

    if not args_to_insert:
        # if nothing to replace... exist early.
        return doc, call_sign

    add_kwargs_param_signature = False
    for name, parameter in call_sign.parameters.items():
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            call_parameters[name] = parameter
        else:
            add_kwargs_param_signature = True

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
                for base in bases:  # don't bother checking `object`
                    if base_arg_dict := getattr(base, "_arg_dict", None):
                        super_doc_dict.update(base_arg_dict)
            if arg[0] != "*":
                # do not process *, *args, or **kwargs parameters
                if class_name == "super":
                    arg_dict = super_doc_dict
                    if arg not in super_doc_dict:
                        raise TypeError(
                            f"Argument {arg} not found in {cls.__name__}'s inheritance."
                        )
                else:
                    # import the class and get it's arg_dict
                    module_name, m_class_name = class_name.rsplit(".", 1)
                    try:
                        target_cls = getattr(
                            importlib.import_module(module_name), m_class_name
                        )
                    except ImportError:
                        raise TypeError(
                            f"Unable to import class {class_name} for docstring replacement."
                        ) from None
                    if target_cls not in bases:
                        raise TypeError(
                            f"{target_cls.__name__} is not a parent of {cls.__name__}"
                        )
                    arg_dict = getattr(target_cls, "_arg_dict", None)
                    if not arg_dict:
                        raise TypeError(
                            f"{target_cls} must have an _arg_dict attribute."
                        )
                    if arg not in arg_dict:
                        raise TypeError(
                            f"{arg}'s description not found in {target_cls}._arg_dict."
                        )
                replacement = arg_dict[arg]
                args.append(arg)
                if arg not in call_parameters:
                    param = replacement["parameter"]
                    call_parameters[arg] = param.replace(
                        kind=inspect.Parameter.KEYWORD_ONLY
                    )
        if args:
            doc = _replace_doc_args(doc, f"%({arg_insert})", args, [replacement])

    star_args_classes = [
        match.group("class_name") for match in REPLACE_STAR_REGEX.finditer(doc)
    ]

    for star_class in star_args_classes:
        if star_class == "super":
            add_kwargs_param_signature = False
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

    if add_kwargs_param_signature:
        call_parameters["kwargs"] = inspect.Parameter(
            "kwargs", inspect.Parameter.VAR_KEYWORD
        )
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

    def __new__(
        mcs,
        name,
        bases,
        namespace,
        star_excludes: Optional[set] = None,
        update_signature: bool = True,
        **kwargs,
    ):
        # construct the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # build the argument dictionary
        if getattr(cls, "__doc__", None) is None:
            # if I don't have a __doc__ don't do anything.
            cls._arg_dict = {}
            return cls

        cls._arg_dict = _class_arg_doc_dict(cls)
        if star_excludes is None:
            star_excludes = set()
        cls.__doc__, init_sig = _doc_replace(cls, star_excludes)
        # Need to look inside __dict___ to check if the class
        # actually has an __init__ defined for it. Can't check
        # for cls.__init__ because it could pull the parent's
        # __init__ function
        init_func = cls.__dict__.get("__init__", None)
        if init_func and update_signature:
            cls.__init__ = bind_signature_to_function(init_sig, init_func)
        return cls


# Could also add this functionality as a wrapper for a class.
