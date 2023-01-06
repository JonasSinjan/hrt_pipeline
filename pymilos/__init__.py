try:
    import sys
    import pymilos
except Exception as e:
    # from .pymilos import *
    print(e)
    print("unable to import pymilos version in __init__.py (this is o.k.)")
