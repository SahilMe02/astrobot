"""Test setup: let modules that call google.auth.default() at import time
load without real credentials. Tests never call Google Cloud; if ADC is
configured (e.g. on a developer laptop) it is left untouched."""

import os

import google.auth
from google.auth import exceptions
from google.auth.credentials import AnonymousCredentials

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

try:
    google.auth.default()
except exceptions.DefaultCredentialsError:
    google.auth.default = lambda *a, **k: (AnonymousCredentials(), os.environ["GOOGLE_CLOUD_PROJECT"])
