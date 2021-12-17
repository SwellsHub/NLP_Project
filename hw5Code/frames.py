import re

class Frame:
    """A simple class for frames.

    type (str): the type of the frame, e.g., 'find-train'
    args (dict): a mapping from argument names to argument values, e.g., 
                   { 'train-departure' : 'Cambridge',
                     'train-destination' : 'Stansted Airport' }
    """
    def __init__(self, type, args):
        self.type = type
        self.args = args

    @staticmethod
    def from_str(s):
        """Convert a string to a frame."""

        # This parsing is VERY brittle and shouldn't be used as an
        # example of good coding practice!
        
        m = re.fullmatch(r'\s*([^\s(]*)\s*\((.*)\)\s*', s)
        if m is None:
            raise ValueError(f"Couldn't convert {repr(s)} to Frame")
        type = m.group(1)
        args = {}
        for argval in m.group(2).split(';'):
            argval = argval.strip()
            if not argval:
                continue
            m = re.fullmatch(r'\s*(\S+)\s*=\s*(.*?)\s*', argval)
            if m is None:
                raise ValueError(f"Couldn't parse argument {repr(argval)} in string {s}")
            arg = m.group(1)
            val = m.group(2)
            args[arg] = val
            
        return Frame(type, args)

    def __repr__(self):
        return f'Frame({repr(self.type)},{repr(self.args)})'

    def __str__(self):
        args = ' ; '.join(f'{arg} = {val}' if val is not None else arg for arg, val in self.args.items())
        return f'{self.type} ( {args} )'

    def __eq__(self, other):
        return (isinstance(other, Frame) and
                self.type == other.type and
                self.args == other.args)

    def __ne__(self, other):
        return not self.__eq__(other)
