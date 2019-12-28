# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

from pathlib import Path
from exawind.prelude import utils

def test_ensure_directory(tmpdir):
    tstdir = tmpdir.mkdir("test_case")
    cname = str(tstdir)
    newdir = utils.ensure_dir(Path(cname) / "test_dir")
    assert newdir.exists()

def test_user_info():
    # Just testing execution for now
    utils.username()
    assert utils.user_home_dir() == Path.home()

def test_abspath():
    stk_root = "${HOME}/exawind/install/gcc/trilinos"
    fpath = utils.abspath(stk_root)
    factual = Path.home() / "exawind" / "install" / "gcc" / "trilinos"
    assert fpath == factual
