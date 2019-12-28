# -*- coding: utf-8 -*-

"""\
Miscellaneous utilities
-----------------------
"""

import os
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import pytz

def username():
    """User's login name on the system"""
    import getpass
    return getpass.getuser()

def user_home_dir():
    """Absolute path to user's home directory"""
    homedir = Path.home()
    if homedir.exists():
        return homedir

    for envvar in ["HOME", "USERPROFILE"]:
        path = os.environ.get(envvar, None)
        if path is not None:
            pth = Path(path)
            if pth.is_dir():
                return pth
    return None

def timestamp(time_format=None, local=False, time_zone=pytz.utc):
    """Return a formatted timestamp

    Args:
        time_format (str): A format suitable for ``strftime``
        local (bool): If True, return timestamp in local timezone
        time_zone: If local is False, returns timestamp in the given timezone

    Return:
        str: A formatted timestamp string
    """
    time_fmt = time_format or "%Y-%m-%d %H:%M:%S (%Z)"
    if local:
        return datetime.now().strftime(time_fmt)
    else:
        return datetime.now(time_zone).strftime(time_fmt)

def backup_filename(fname, time_format=None, time_zone=pytz.utc):
    """Return a backup filename with timestamp

    Args:
        time_format (str): A format suitable for ``strftime``
        time_zone: If local is False, returns timestamp in the given timezone

    Return:
        str: A suitable backup filename with timestamp information
    """
    time_fmt = time_format or "%Y%m%d-%H%M%S-%Z"
    local = time_zone is None
    tstamp = timestamp(time_fmt, local=local, time_zone=time_zone)
    pname = Path(fname)
    new_name = pname.stem + "_" + tstamp + pname.suffix
    return pname.parent / new_name

def abspath(fpath):
    """Return the absolute path of a given file/directory

    Differs from functions in standard library n that this function will expand
    home directory and shell variables, i.e., combines ``expanduser``,
    ``expandvars`` and ``abspath`` in one call,

    Args:
        fpath (str): Path name or Path-like object

    Return:
        Path: Absolute path
    """
    ptmp1 = os.path.expanduser(fpath)
    ptmp2 = os.path.expandvars(ptmp1)
    return Path(ptmp2).resolve()

def ensure_dir(dpath):
    """Ensure that a directory exists"""
    absdir = Path(dpath)
    if not absdir.exists():
        absdir.mkdir(parents=True)
    return absdir

@contextmanager
def exec_dir(dpath, create=False):
    """Execute code from within a given directory

    The original working directory is restored after the execution of the block

    .. code-block:: python
       with exec_dir("../../tests") as wdir:
           with open("test_file.dat", 'r') as fh:
               lines = fh.readlines()
    """
    absdir = Path(dpath).resolve()
    if create and not absdir.exists():
        absdir.mkdir(parents=True)

    pwd = Path.cwd()
    try:
        os.chdir(absdir)
        yield absdir
    finally:
        os.chdir(pwd)
