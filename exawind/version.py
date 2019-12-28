# -*- coding: utf-8 -*-

"""
Exawind version information
"""

import os
import subprocess
import shlex
from pathlib import Path

__all__ = [
    "version",
    "full_version",
    "git_revision"
]

version = "v0.0.1"

def get_git_revision():
    """Return the git SHA ID if available

    Adapted from: https://github.com/numpy/numpy/blob/master/setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        task = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env)
        return task


    dirname = Path(__file__).parent.resolve()
    cwd = os.getcwd()
    git_ver = "unknown"
    try:
        os.chdir(dirname)
        cmdline = "git rev-parse HEAD"
        cmd = shlex.split(cmdline)
        task = _minimal_ext_cmd(cmd)
        out, _ = task.communicate()
        if task.poll() == 0:
            git_ver = out.strip().decode('ascii')
    finally:
        os.chdir(cwd)

    return git_ver

git_revision = get_git_revision()

full_version = version
if git_revision != "unknown":
    full_version = version + "-g" + git_revision[:7]
