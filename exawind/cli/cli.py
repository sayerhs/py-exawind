# -*- coding: utf-8 -*-

"""\
ExaWind Command Line Interface (CLI)
------------------------------------
"""

import logging
import argparse
import shlex
from ..config.config import config_manager, ExawindConfig
from ..version import full_version

_lgr = logging.getLogger(__name__)

def get_epilog():
    """Return the epilog string"""
    return "ExaWind Python Library %s"%full_version

class CLIBase:
    """ExaWind CLI interface

    This class defines a basic interface for all CLI applications derived from
    ExaWind Python Library.
    """

    #: Banner
    banner = "ExaWind Python Library %s"%full_version

    #: Description of the CLI app used in help messages
    description = "ExaWind application"
    #: Epilog for help messages
    epilog = banner

    #: Options controlling verbosity of the scripts
    app_levels = ["INFO", "DEBUG"]
    #: Options controlling verbosity of the library
    lib_levels = ["WARNING", "INFO", "DEBUG"]

    def __init__(self, name=None, args=None):
        """
        Args:
            name (str): Custom name used in help messages for the application
            args (str): Arguments to be passed to argparse instead of sys.argv
        """
        #: Name of the application
        self.name = name
        #: Parser for command line arguments
        self.parser = argparse.ArgumentParser(
            description=self.description,
            epilog=self.epilog,
            prog=name)
        #: Configuration object
        self.cfg = ExawindConfig()

        # Initialize command line arguments
        self.cli_options()

        # Parse arguments
        if args:
            self.args = self.parser.parse_args(shlex.split(args))
        else:
            self.args = self.parser.parse_args()

    def cli_options(self):
        """Setup common CLI options for all apps"""
        parser = self.parser
        parser.add_argument(
            '-V', '--version', action='version',
            version=self.banner)
        verbosity = parser.add_mutually_exclusive_group(required=False)
        verbosity.add_argument(
            '--quiet', action='store_true',
            help="Quiet mode; disable all informational messages")
        verbosity.add_argument(
            '-v', '--verbose', action='count', default=0,
            help="Increase verbosity of messages. Default: off")

    def __call__(self):
        """Execute the CLI application"""
        args = self.args
        self.setup_logging(args.verbose, args.quiet)

    def setup_logging(self, verbosity=0, quiet=False):
        """Setup logging based on command line options"""
        app_levels = self.app_levels
        lib_levels = self.lib_levels
        cfgmgr = config_manager()
        cfg = cfgmgr.get_config(init_logging=False)
        log_cfg = cfg.exawind.logging
        lcfg = log_cfg.pylogger_options
        if quiet:
            lcfg.handlers.exawind_console.level = "ERROR"
            lcfg.handlers.exawind_script_console.level = "ERROR"
        else:
            alevel = app_levels[min(verbosity, len(app_levels) -1)]
            llevel = lib_levels[min(verbosity, len(lib_levels) -1)]
            lcfg.handlers.exawind_console.level = llevel
            lcfg.handlers.exawind_script_console.level = alevel
        cfgmgr.configure_logging(log_cfg)

        _lgr.info(self.banner)
        rcfiles = cfgmgr.cfg_files
        msg = ("Config files loaded: %s"%rcfiles
               if rcfiles else
               "No configuration found; using defaults")
        _lgr.debug(msg)
        self.cfg = cfg

class CLISubCmdBase(CLIBase):
    """An app with sub-commands"""

    def __init__(self, *args, **kwargs):
        self.sub_commands = {}
        super().__init__(*args, **kwargs)

    def cli_options(self):
        """Setup the sub-parsers"""
        super().cli_options()
        self.subparsers = self.parser.add_subparsers(
            description="Available sub-commands; use -h to see additional information",
            metavar="command")

    def __call__(self):
        """Execute the sub-command"""
        super().__call__()
        self.args.func(self.args)
        _lgr.info(self.banner)
