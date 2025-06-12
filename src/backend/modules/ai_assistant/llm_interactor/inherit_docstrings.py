class InheritDocstrings(type):
    """
    AI-generated code that inherits docstrings from base classes.

    Usage:
    class CLASS_NAME(OTHER_INHERITANCE, metaclass=InheritDocstrings):
    """

    def __new__(cls, name, bases, dct):
        for attr, obj in dct.items():
            for base in bases:
                if attr in base.__dict__ and obj.__doc__ is None:
                    obj.__doc__ = base.__dict__[attr].__doc__
        return super().__new__(cls, name, bases, dct)
