# -*- coding: utf-8 -*-

"""\
Nalu-Wind Tasks CLI
--------------------
"""

import logging
from pathlib import Path
from ...prelude.struct import Struct
from ...nalu.task import NaluTaskRunner, NaluTask

_lgr = logging.getLogger(__name__)

class NaluTaskCLI:
    """Nalu-Wind Tasks sub-command"""
    # pylint: disable=too-few-public-methods

    def __init__(self, parser):
        """
        Args:
            parser: An argparse instance
        """
        self.parser = parser
        parser.add_argument(
            '-l', '--list-tasks', action='store_true',
            help="List available tasks and exit")
        parser.add_argument(
            '-i', '--input-file', default='nalu_tasks.yaml',
            help="Input file describing pre/post tasks (nalu_tasks.yaml)")
        parser.set_defaults(func=self)

    def run_tasks(self, args):
        """Execute the tasks"""
        inpfile = args.input_file
        fpath = Path(inpfile).resolve()
        if not fpath.exists():
            _lgr.error("Input file not found: %s. Exiting!", inpfile)
            self.parser.exit(1)
        opts = Struct.load_file(fpath)
        if "nalu_tasks" not in opts:
            _lgr.error("Cannot find nalu_tasks entry in input file")
            self.parser.exit(1)
        _lgr.info("Executing nalu-wind tasks from file: %s", inpfile)
        runner = NaluTaskRunner()
        runner(opts.nalu_tasks)
        _lgr.info("All tasks completed successfully")


    def __call__(self, args):
        """Execution actions"""
        if args.list_tasks:
            print("Available tasks: ")
            for key, value in NaluTask.available_tasks().items():
                full_doc = value.__doc__
                doc = full_doc.split("\n")[0].strip()
                print("- %s :: %s"%(key, doc))
            print()
            return

        self.run_tasks(args)
