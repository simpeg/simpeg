import inspect
import itertools
import re
import collections
import importlib

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


def _parse_numpydoc_parameters(doc):
    doc_sections = NUMPY_SECTION_REGEX.search(doc).groupdict()
    parameters = doc_sections.get("parameters")
    if parameters is None:
        parameters = ""

    others = doc_sections.get("other_parameters")
    if others is not None:
        parameters += "\n" + others
    return parameters


def _class_arg_doc_dict(cls):
    doc = inspect.cleandoc(cls.__doc__)
    parameters = _parse_numpydoc_parameters(doc)

    arg_dict = collections.OrderedDict()
    init_params = inspect.signature(cls.__init__).parameters
    for match, next_match in _pairwise(NUMPY_ARG_TYPE_REGEX.finditer(parameters)):
        arg, type_string = match.groups()

        # skip over values that are to be replaced (if there are any left)
        # and skip over **kwargs
        if not REPLACE_REGEX.match(arg) and arg != "**kwargs":
            # +1 removes the newline character at the end of the argument : type_name match
            start = match.end() + 1
            # -1 removes the newline character at the start of the argument : type_name match (if there was one).
            end = next_match.start() - 1 if next_match is not None else None
            description = parameters[start:end]
            if description == "":
                description = None
            for arg_part in ARG_SPLIT_REGEX.split(arg):
                arg_dict[arg_part] = {
                    "type_string": type_string,
                    "description": description,
                    "parameter": init_params.get(arg, None),
                }
    return arg_dict


def _doc_replace(cls, star_excludes):
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

    add_kwargs_param_signature = False
    for name, parameter in call_sign.parameters.items():
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            call_parameters[name] = parameter
        else:
            add_kwargs_param_signature = True
    bases = cls.__mro__[1:]

    for arg_insert in args_to_insert:
        args = []
        type_string = None
        desc_string = None
        for item in ARG_SPLIT_REGEX.split(arg_insert):
            class_name, arg = item.rsplit(".", 1)
            if arg[0] != "*":
                # do not process *, *args, or **kwargs parameters
                target_cls = None
                if class_name == "super":
                    for base in bases[:-1]:  # Don't bother checking `object`
                        if issubclass(base, BaseSimPEG):
                            if arg in base._arg_dict:
                                target_cls = base
                                break
                    if not target_cls:
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
                            f"Unable to import class {class_name} for docstring replacement"
                        )
                if (
                    not issubclass(target_cls, BaseSimPEG)
                    or arg not in target_cls._arg_dict
                ):
                    raise TypeError(f"Argument {arg} not found in {target_cls}.")
                replacement = target_cls._arg_dict[arg]
                if arg not in call_parameters:
                    param = inspect.signature(target_cls.__init__).parameters[arg]
                    call_parameters[arg] = inspect.Parameter(
                        param.name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=param.default,
                        annotation=param.annotation,
                    )
                args.append(arg)
                type_string = replacement["type_string"]
                desc_string = replacement["description"]

        if args:
            replace_string = ", ".join(args)
            if type_string:
                replace_string += f" : {type_string}"
            if desc_string:
                replace_string += "\n" + desc_string
            doc = doc.replace(f"%({arg_insert})", replace_string)

    star_args_classes = [
        match.group("class_name") for match in REPLACE_STAR_REGEX.finditer(doc)
    ]

    for star_class in star_args_classes:
        star_arg_dict = collections.OrderedDict()
        if star_class == "super":
            add_kwargs_param_signature = False
            # Grab everything from __mro__ that isn't already in signature.
            bases = cls.__mro__[1:]
            for base in bases[:-1]:
                if issubclass(base, BaseSimPEG):
                    star_arg_dict.update(base._arg_dict)
        else:
            module_name, class_name = star_class.rsplit(".", 1)
            try:
                target_cls = getattr(importlib.import_module(module_name), class_name)
            except ImportError:
                raise ImportError(
                    f"Unable to import class {class_name} for docstring replacement"
                )
            if not issubclass(target_cls, BaseSimPEG):
                raise TypeError(f"{target_cls} must be a subclass of BaseSimPEG.")
            star_arg_dict = target_cls._arg_dict

        rep_string_parts = []
        for arg, replacement in star_arg_dict.items():
            if arg not in star_excludes and arg not in call_parameters:
                replace_string = arg
                if replacement["type_string"]:
                    replace_string += f" : {replacement['type_string']}"
                if replacement["description"]:
                    replace_string += "\n" + replacement["description"]
                rep_string_parts.append(replace_string)
                param = replacement["parameter"]
                call_parameters[arg] = inspect.Parameter(
                    param.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=param.default,
                    annotation=param.annotation,
                )
        rep_string = "\n".join(rep_string_parts)
        doc = doc.replace(f"%({star_class}.*)", rep_string)

    if add_kwargs_param_signature:
        call_parameters["kwargs"] = inspect.Parameter(
            "kwargs", inspect.Parameter.VAR_KEYWORD
        )
    new_signature = inspect.Signature(parameters=call_parameters.values())
    return doc, new_signature


class DoceratorMeta(type):

    def __new__(mcs, name, bases, namespace, star_excludes=None):
        # construct the class
        cls = super().__new__(mcs, name, bases, namespace)
        # build the argument dictionary
        if getattr(cls, "__doc__", None) is None:
            cls._arg_dict = {}
            return cls

        cls.__doc__ = inspect.cleandoc(cls.__doc__)
        cls._arg_dict = _class_arg_doc_dict(cls)
        if star_excludes is None:
            star_excludes = []
        cls.__doc__, init_sig = _doc_replace(cls, star_excludes)
        # check if the class actually has an __init__ defined for it
        # can't do getattr(cls, "__init__") here because it could
        # pull the parent's __init__ function
        init_func = cls.__dict__.get("__init__", None)
        if init_func:

            def init_bind_to_signature(*args, **kwargs):
                try:
                    params = init_sig.bind(*args, **kwargs)
                except TypeError as err:
                    raise TypeError(f"{cls.__name__}.__init__(): {err}") from None
                return init_func(*params.args, **params.kwargs)

            init_bind_to_signature.__signature__ = init_sig

            cls.__init__ = init_bind_to_signature
        return cls


class BaseSimPEG(metaclass=DoceratorMeta):
    """Base class for simpeg classes."""

    # Developer note:
    # This class is mostly used to identify simpeg classes
    # and to catch any leftover keyword arguments before calling
    # object.__init__() (if that was the next class on the mro above
    # this one). If there are any leftover arguments, it throws a TypeError
    # with an appropriate message reference the class that was initialized.
