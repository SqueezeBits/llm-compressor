import os


ENFORCE_EAGER = os.getenv("RBLN_COMPILE", "0") == "0"
USE_CUSTOM_OPS = os.getenv("USE_CUSTOM_OPS", "0") == "1"
