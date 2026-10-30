import importlib

# Alias for 'iam-digits'
_module = importlib.import_module('.iam-digits', package=__package__)
Source = _module.Source

