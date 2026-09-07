"""Make the sweep scripts find the packages in this archive's layout.

Each sweep puts its own folder on sys.path; in the archive the packages sit one
level up.  Either import this module first, or run them as

    PYTHONPATH=.. LACF_UNITS=cost python3 sliver_referral.py --views ...
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
