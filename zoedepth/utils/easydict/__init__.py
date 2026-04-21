"""
EasyDict implementation for nested dictionary access.
"""

from collections import OrderedDict


class EasyDict(dict):
    """
    EasyDict: Extended dictionary that allows attribute-style access.
    Based on dash's EasyDict implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v if isinstance(v, dict) else EasyDict(v) if isinstance(v, dict) else v
            elif isinstance(arg, list):
                for i, v in enumerate(arg):
                    self[str(i)] = v if isinstance(v, dict) else EasyDict(v) if isinstance(v, dict) else v
        
        for k, v in kwargs.items():
            self[k] = v if isinstance(v, dict) else EasyDict(v) if isinstance(v, dict) else v
    
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value):
        self[name] = value if not isinstance(value, dict) else EasyDict(value)
    
    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)
    
    def __iter__(self):
        return iter(dict.items(self))
    
    def __reversed__(self):
        return reversed(list(dict.keys(self)))
    
    def __contains__(self, name) -> bool:
        return dict.__contains__(self, name) or (hasattr(self, name) and name not in dir(self))