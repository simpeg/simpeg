import re
import inspect
import itertools
import importlib
from ..props import BaseSimPEG

__all__ = ["class_arg_doc_dict", "doc_inherit"]

REPLACE_REGEX = re.compile(r"%\((?P<replace_key>.*)\)")
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

KV_REGEX = re.compile(r"^[^\s].*$", flags=re.M)
NUMPY_SECTION_REGEX = re.compile(
    rf"^(?P<summary>[\s\S]+?)??{'?'.join(_section_regexs)}?$"
)
ARG_TYPE_SEP_REGEX = re.compile(r"\s*:\s*")
ARG_SPLIT_REGEX = re.compile(r"\s*,\s*")
NUMPY_ARG_TYPE_REGEX = re.compile(
    r"^(?P<arg_name>\S.*?)(?:\s*:\s*(?P<type>.*?))?$", re.MULTILINE
)

__cached_class_arg_dicts = {}


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


def class_arg_doc_dict(this_class):
    if not issubclass(this_class, BaseSimPEG):
        # don't do anything to classes not from simpeg
        return {}
    if this_class.__doc__ is None:
        return {}
    if __cached_class_arg_dicts.get(this_class, None) is not None:
        return __cached_class_arg_dicts[this_class]

    doc = inspect.cleandoc(this_class.__doc__)
    parameters = _parse_numpydoc_parameters(doc)

    # get the classes call signature:
    call_signature = inspect.signature(this_class.__init__)

    arg_dict = {}
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
            arg_dict[arg] = {
                "type_string": type_string,
                "description": description,
                "parameters": [
                    call_signature.parameters[parg]
                    for parg in ARG_SPLIT_REGEX.split(arg)
                ],
            }
    # cache this for future lookups
    __cached_class_arg_dicts[this_class] = arg_dict
    return arg_dict


def doc_inherit(star_excludes=None):
    if star_excludes is None:
        star_excludes = set()
    else:
        star_excludes = set(star_excludes)

    def doc_decorator(this_class):
        # a list of arguments and the class they would resolve to.
        # search through my docstring and figure out what to replace.
        doc = inspect.cleandoc(this_class.__doc__)
        # replacement items in doc
        args_to_insert = [
            match.group("replace_key").rsplit(".", 1)
            for match in REPLACE_REGEX.finditer(doc)
        ]

        # grab the current call signature parameters
        # will append items insert using the replacement.
        call_signature_parameters = []
        call_signature_arg_names = []
        add_kwargs_param_signature = False
        this_call_sign = inspect.signature(this_class.__init__)
        for name, parameter in this_call_sign.parameters.items():
            if parameter.kind != inspect.Parameter.VAR_KEYWORD:
                call_signature_arg_names.append(name)
                call_signature_parameters.append(parameter)
            else:
                add_kwargs_param_signature = True

        super_doc_dict = None
        do_star_replace = False
        for class_name, arg in args_to_insert:
            if class_name == "super" and super_doc_dict is None:
                # build the super_doc_dict if it will be needed.
                super_doc_dict = {}
                for cls in this_class.__mro__[1:-1]:
                    cls_doc_dict = class_arg_doc_dict(cls)
                    super_doc_dict = cls_doc_dict | super_doc_dict
            if arg != "*":
                if class_name == "super":
                    replacement = super_doc_dict[arg]
                else:
                    # import the class and get it's arg_dict
                    module_name, m_class_name = class_name.rsplit(".", 1)
                    target_cls = getattr(
                        importlib.import_module(module_name), m_class_name
                    )
                    replacement = class_arg_doc_dict(target_cls)[arg]
                # build the replacement string
                replace_string = arg
                if replacement["type_string"] is not None:
                    replace_string += " : " + replacement["type_string"]
                if replacement["description"] != "":
                    replace_string += "\n" + replacement["description"]
                doc = doc.replace(f"%({class_name}.{arg})", replace_string)
                for param in replacement["parameters"]:
                    if param.name not in call_signature_arg_names:
                        # This meant it was supposed to be caught by **kwargs.
                        # and must be a KEYWORD_ONLY type parameter.
                        # We create a new `Parameter`, because we can't change the
                        # `kind` of a previously created `Parameter`.
                        call_signature_arg_names.append(param.name)
                        call_signature_parameters.append(
                            inspect.Parameter(
                                param.name,
                                inspect.Parameter.KEYWORD_ONLY,
                                default=param.default,
                                annotation=param.annotation,
                            )
                        )
            else:
                do_star_replace = True

        if do_star_replace:
            parameters = _parse_numpydoc_parameters(doc)

            exclusions = set()
            for match in NUMPY_ARG_TYPE_REGEX.finditer(parameters):
                # this also matches the replacement regex, but that
                # should never match anything else anyways...
                arg = match.group("arg_name")
                exclusions.add(arg)
            exclusions.update(star_excludes)

            # search through again, but this time only deal with the "*" replacements.
            # add any added items to the exclusions and assume the order things are listed
            # in the documentation represents the precedence of "*" replacements.
            for class_name, arg in args_to_insert:
                if arg == "*":
                    star_arg_dict = None
                    if class_name == "super":
                        star_arg_dict = super_doc_dict
                        # if we're getting everything, don't add **kwargs to the signature?
                        add_kwargs_param_signature = False
                    else:
                        # import the class and get it's arg_dict
                        module_name, class_name = class_name.rsplit(".", 1)
                        target_cls = getattr(
                            importlib.import_module(module_name), class_name
                        )
                        star_arg_dict = class_arg_doc_dict(target_cls)
                    # grab everything in star_arg_dict
                    replacements = []
                    for s_arg, replacement in star_arg_dict.items():
                        if s_arg not in exclusions:
                            replace_string = s_arg
                            if replacement["type_string"] is not None:
                                replace_string += " : " + replacement["type_string"]
                            if replacement["description"] != "":
                                replace_string += "\n" + replacement["description"]
                            replacements.append(replace_string)
                            exclusions.add(s_arg)
                            for param in replacement["parameters"]:
                                if param.name not in call_signature_arg_names:
                                    # This meant it was supposed to be caught by **kwargs.
                                    # and must be a KEYWORD_ONLY type parameter.
                                    # We create a new `Parameter`, because we can't change the
                                    # `kind` of a previously created `Parameter`.
                                    call_signature_arg_names.append(param.name)
                                    call_signature_parameters.append(
                                        inspect.Parameter(
                                            param.name,
                                            inspect.Parameter.KEYWORD_ONLY,
                                            default=param.default,
                                            annotation=param.annotation,
                                        )
                                    )

                    replace_string = "\n".join(replacements)
                    doc = doc.replace(f"%({class_name}.{arg})", replace_string)
        if add_kwargs_param_signature:
            call_signature_parameters.append(
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            )

        this_class.__doc__ = doc
        this_class.__init__.__signature__ = inspect.Signature(
            parameters=call_signature_parameters
        )
        return this_class

    return doc_decorator
