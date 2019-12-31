# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""\
Nalu-Wind Task
--------------
"""

import importlib
import logging
import abc
import glob
from pathlib import Path
import stk
from .cfd_mesh import CFDMesh

_lgr = logging.getLogger(__name__)

class NaluTask(abc.ABC):
    """Abstract representation of a task"""

    #: Mapping of available tasks registered
    _available_tasks = dict()

    @classmethod
    def create(cls, task_name, cfd_mesh):
        """Create an instance of the subclass based on class registration"""
        if task_name not in cls._available_tasks:
            raise KeyError("Invalid task name: %s"%task_name)
        tcls = cls._available_tasks[task_name]
        obj = tcls(cfd_mesh)
        return obj

    def __init__(self, mesh):
        self.mesh = mesh

    def __init_subclass__(cls, task_name, **kwargs):
        """Register a task"""
        if task_name in cls._available_tasks:
            raise KeyError("Task name = %s already registered with %s"%(
                task_name, cls._available_tasks[task_name].__name__))
        super().__init_subclass__(**kwargs)
        cls._available_tasks[task_name] = cls
        cls._task_name = task_name

    def process_inputs(self, *args, **kwargs):
        """Parse inputs"""

    def init_meta_data(self):
        """Perform initialization actions"""

    def names_to_parts(self, pnames, must_exist=True):
        """Convert part names to parts"""
        meta = self.mesh.meta
        for pname in pnames:
            part = meta.get_part(pname, must_exist)
            yield part

    @abc.abstractmethod
    def run(self):
        """Perform the task"""


class NaluTaskRunner:
    """Orchestrate running the tasks"""

    def __init__(self, cfd_mesh=None, par=None):
        """
        Args:
            inputs (Struct): Inputs for the tasks to be executed
            cfd_mesh (CFDMesh): Mesh instance if available
            par (stk.Parallel): Parallel instance to create ``cfd_mesh``
        """
        if cfd_mesh is not None:
            self.mesh = cfd_mesh
        else:
            par_obj = par or stk.Parallel.initialize()
            self.mesh = CFDMesh(par_obj)

        #: List of tasks to be executed
        self.task_list = []
        #: Task configuration info object
        self.inputs = None
        #: Default parts for all tasks if available
        self.parts = None
        #: Time at which the solution fields are read from file
        self.read_time = 0.0

    def __call__(self, inputs):
        """Initialize and run all tasks"""
        self.inputs = inputs
        self.process_inputs()
        self.init_mesh()
        mesh_modified = self.run_tasks()
        self.write_output_mesh(mesh_modified)

    def process_inputs(self):
        """Process input dictionary"""
        inputs = self.inputs
        mesh = self.mesh
        assert "tasks" in inputs, "Cannot find 'tasks' definition in inputs"

        self.parts = inputs.get("parts", None)

        tasks = inputs.tasks
        task_list = []
        for task in tasks:
            task_name = next(iter(task))
            task_inp = task[task_name]
            tobj = NaluTask.create(task_name, mesh)
            tobj.process_inputs(task_inp, self.parts)
            task_list.append(tobj)
        self.task_list = task_list

    def init_mesh(self):
        """Initialize the mesh metadata"""
        inputs = self.inputs
        read_mesh = "input_db" in inputs
        if read_mesh:
            _lgr.info("NaluTaskRunner: initializing mesh meta data")
            self.mesh.init_mesh_meta(inputs.input_db)

        for task in self.task_list:
            task.init_meta_data()

        read_time = 0.0
        if read_mesh:
            _lgr.info("NaluTaskRunner: populating bulk data")
            read_time = self.mesh.init_mesh_bulk(inputs.input_db)
        else:
            self.mesh.meta.commit()
        self.read_time = read_time

    def run_tasks(self):
        """Execute the given tasks"""
        mesh_modified = False
        for task in self.task_list:
            task_mod = task.run()
            mesh_modified = mesh_modified or task_mod
        return mesh_modified

    def write_output_mesh(self, mesh_modified):
        """Write out the modified mesh"""
        inputs = self.inputs
        if not "output_db" in inputs:
            if mesh_modified:
                _lgr.warning("NaluTaskRunner: mesh was modified but no output_db provided")
            return

        out_opts = inputs.output_db
        force_write = out_opts.get("force_write", False)
        if not (force_write or mesh_modified):
            return

        self.mesh.write_output_db(out_opts, self.read_time)

def _load_defined_tasks():
    """Auto-load and register defined tasks"""
    task_path = Path(__file__).parent.resolve() / "nalu_tasks"
    py_files = glob.glob(str(task_path / "[a-z]*.py"))
    modset = {Path(ff).stem for ff in py_files}
    for pymod in modset:
        importlib.import_module(".%s"%pymod, 'exawind.nalu.nalu_tasks')

_load_defined_tasks()
