# -*- coding: utf-8 -*-

"""\
Nalu-Wind utilities CLI
-----------------------
"""

from ..cli import CLISubCmdBase
from .tasks_cli import NaluTaskCLI

class NaluCLI(CLISubCmdBase):
    """Nalu-Wind utilities command line application"""

    description = "ExaWind Nalu-Wind Utilities"

    def cli_options(self):
        """Setup sub-commands for Nalu-Wind utilities"""
        super().cli_options()
        subparsers = self.subparsers

        tasks = subparsers.add_parser(
            "tasks",
            description="Run pre/post-processing tasks as defined in YAML file",
            help="run pre/post processing tasks")
        self.sub_commands["tasks"] = NaluTaskCLI(tasks)


def main():
    """Execute exawind_nalu"""
    cmd = NaluCLI()
    cmd()
