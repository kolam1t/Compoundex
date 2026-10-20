# PythonAnywhere WSGI configuration
# Paste the contents of this file into the WSGI configuration file that
# PythonAnywhere creates for your web app (Web tab -> WSGI configuration file),
# replacing everything already there.
# Configured for PythonAnywhere username: arsenddnv

import sys

path = '/home/arsenddnv/compoundex'
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application  # noqa
