import inspect
from inspect import Parameter
import pytest
import simpeg.base.doc_inherit as doc_inherit


class Parent(metaclass=doc_inherit.DoceratorMeta):
    """A docstring

    Parameters
    ----------
    arg1 : object
        Extended Description.
    arg2 : int
        2 Extended Description.
    arg3 : int
        3 Extended Description.

    Other Parameters
    ----------------
    even_more : list
    but_not_too_much
        But another description.
    """

    def __init__(self, arg1, arg2, arg3, even_more, but_not_too_much): ...


class ChildClass(Parent):
    __doc__ = f"""Docstring

    Parameters
    ----------
    arg1 : int
        Not quite the same as parent
    a_new_arg : dict
        A dictionary.
    %({Parent.__module__}.{Parent.__qualname__}.arg2)

    Other Parameters
    ----------------
    %(super.*)
    """

    def __init__(self, arg1, a_new_arg, **kwargs): ...


class GrandchildClass(
    ChildClass,
    star_excludes={
        "but_not_too_much",
    },
):
    __doc__ = f"""Docstring

    Parameters
    ----------
    %({Parent.__module__}.{Parent.__qualname__}.arg1)
    a_new_arg : dict
        Still a dictionary..
    %(super.arg2)

    Other Parameters
    ----------------
    %({Parent.__module__}.{Parent.__qualname__}.*)
    """

    def __init__(self, arg1, a_new_arg, **kwargs): ...


@pytest.mark.parametrize("args", [[], ["single"], ["two", "args"]])
def test_doc_replace(args):
    docstring = """A docstring
    Parameters
    ----------
    %(replace.me)
    """
    info = [{"type_string": "object", "description": "    Description\n    and more."}]
    verified = f"""A docstring
    Parameters
    ----------
    {', '.join(args)} : object
        Description
        and more.
    """

    if args:
        # this should also be able to test if the indentation level is correct.
        assert (
            doc_inherit._replace_doc_args(docstring, "%(replace.me)", args, info)
            == verified
        )
    else:
        with pytest.raises(ValueError, match="(Incompatible length of infos).*"):
            out = doc_inherit._replace_doc_args(docstring, "%(replace.me)", args, info)
            print(out)


def test_do_nothing():
    class Undocced(metaclass=doc_inherit.DoceratorMeta):
        def __init__(self): ...

    assert Undocced.__doc__ is None


def test_nothing_to_insert():
    docstring = """A docstring

    Parameters
    ----------
    arg1 : object
        Extended Description.
    arg2 : int
        2 Extended Description.
    arg3 : int
        3 Extended Description.

    Other Parameters
    ----------------
    even_more : list
    but_not_too_much
        But another description.
    """

    init_sig = inspect.Signature(
        parameters=[
            Parameter(name="self", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg1", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg2", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg3", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="even_more", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="but_not_too_much", kind=Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    assert (Parent.__doc__, inspect.signature(Parent.__init__)) == (docstring, init_sig)


def test_child_docerator_meta():
    docstring = """Docstring

    Parameters
    ----------
    arg1 : int
        Not quite the same as parent
    a_new_arg : dict
        A dictionary.
    arg2 : int
        2 Extended Description.

    Other Parameters
    ----------------
    arg3 : int
        3 Extended Description.

    even_more : list
    but_not_too_much
        But another description.
    """

    init_sig = inspect.Signature(
        parameters=[
            Parameter(name="self", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg1", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="a_new_arg", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg2", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="arg3", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="even_more", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="but_not_too_much", kind=Parameter.KEYWORD_ONLY),
        ]
    )

    assert (ChildClass.__doc__, inspect.signature(ChildClass.__init__)) == (
        docstring,
        init_sig,
    )


def test_grandchild_docerator_meta():
    docstring = """Docstring

    Parameters
    ----------
    arg1 : object
        Extended Description.
    a_new_arg : dict
        Still a dictionary..
    arg2 : int
        2 Extended Description.

    Other Parameters
    ----------------
    arg3 : int
        3 Extended Description.

    even_more : list
    """

    init_sig = inspect.Signature(
        parameters=[
            Parameter(name="self", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg1", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="a_new_arg", kind=Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(name="arg2", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="arg3", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="even_more", kind=Parameter.KEYWORD_ONLY),
            Parameter(name="kwargs", kind=Parameter.VAR_KEYWORD),
        ]
    )

    assert (GrandchildClass.__doc__, inspect.signature(GrandchildClass.__init__)) == (
        docstring,
        init_sig,
    )
