"""Implement the ES-MDA algorithms."""
__author__ = """Antoine Collet"""
__email__ = "antoine.collet5@gmail.com"
__version__ = "0.3.0"

from .esmda import ESMDA
from .esmda_rs import ESMDA_RS

__all__ = ["ESMDA", "ESMDA_RS"]
