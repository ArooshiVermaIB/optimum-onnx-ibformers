import os
import sys

pth, _ = os.path.split(__file__)

if pth not in sys.path:
    sys.path.append(pth)
