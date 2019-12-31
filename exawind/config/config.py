# -*- coding: utf-8 -*-
# pylint: disable=too-many-ancestors

"""\
Exawind configuration
"""

import sys
import logging
from logging.config import dictConfig
from .. import version
from ..prelude.struct import Struct
from ..prelude import utils
from ..prelude import cfg

try:
    from mpi4py import MPI
    have_mpi = True
except ImportError:
    have_mpi = False

_config_banner = """\
# -*- mode: yaml -*-
#
# Exawind Python Library %(version)s
#
# Auto-generated on: %(timestamp)s
#

"""

class ExawindConfig(Struct):
    """Exawind configuration dictionary"""

    def write_config(self, fh=sys.stdout):
        """Dump current configuration"""
        fh.write(_config_banner%dict(
            version=version.full_version,
            timestamp=utils.timestamp()))
        self.to_yaml(fh)
        fh.write("\n\n")

class ExawindCfgManager(cfg.ConfigManager):
    """Configuration manager object for Exawind Python Library"""

    @staticmethod
    def rc_type():
        """Type of configuration file"""
        return "yaml"

    @staticmethod
    def rc_base():
        """Base name for configuration"""
        return "exawind"

    @staticmethod
    def cfg_class():
        """Configuration object class"""
        return ExawindConfig

    @staticmethod
    def configure_logging(log_cfg=None):
        """Configure python logging"""
        if log_cfg is None:
            logging.basicConfig()
        else:
            logger_cfg = log_cfg.pylogger_options
            if have_mpi:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                if rank > 0:
                    handlers = logger_cfg.handlers
                    handlers.exawind_console.level = "ERROR"
                    handlers.exawind_script_console.level = "ERROR"
                    fmt_opts = logger_cfg.formatters
                    stdout_fmt = fmt_opts.stdout.format
                    fmt_opts.stdout.format = "[%d] "%rank + stdout_fmt
                    default_fmt = fmt_opts.default.format
                    fmt_opts.default.format = "[%d] "%rank + default_fmt
            dictConfig(logger_cfg)


_cfg_manager = cfg.make_config_manager(ExawindCfgManager)

def config_manager():
    """Configuration manager object"""
    return _cfg_manager()

def config():
    """Configuration object"""
    cfgman = _cfg_manager()
    exwcfg = cfgman.get_config().get(cfgman.cfg_root, cfgman.cfg_class()())
    return exwcfg
