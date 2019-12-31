# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-ancestors

"""\
CFD Mesh
--------
"""

import logging
from pathlib import Path
import glob
import numpy as np
import stk
from stk import StkMesh, StkRank
from ..prelude.struct import Struct

_lgr = logging.getLogger(__name__)

class FieldDefinitions(Struct):
    """Field definitions for common fields used in CFD"""

    def add_field_definition(
            self, name,
            ftype="scalar", dtype=np.double,
            rank=StkRank.NODE_RANK,
            num_states=1):
        """Add a new field definition"""
        if name in self:
            raise KeyError("Redefinition of a field is not allowed")
        self[name] = FieldDefinitions(
            name=name,
            ftype=ftype,
            dtype=dtype,
            rank=rank,
            num_states=num_states)

def _create_field_definitions(ndim=3):
    """Helper to create field definitions"""
    def _get_rank(frank):
        """Get the rank"""
        rank_map = {
            1: StkRank.NODE_RANK,
            2: StkRank.EDGE_RANK,
            3: StkRank.FACE_RANK,
            "NODE": StkRank.NODE_RANK,
            "EDGE": StkRank.EDGE_RANK,
            "ELEM": StkRank.ELEM_RANK
        }
        if frank == "SIDE":
            return rank_map[ndim]
        return rank_map[frank]

    dtype_map = {
        "double": np.double,
        "int": np.int,
        "longlong": np.longlong,
        "ulonglong": np.ulonglong,
    }
    fdefs = FieldDefinitions()
    fdir = Path(__file__).parent.resolve()
    ffile = fdir / "field_defs.txt"
    with open(ffile, 'r') as fh:
        for line in fh:
            (name, ftype, dtype, nstates, frank) = [x.strip() for x in line.strip().split()]
            rank = _get_rank(frank.upper())
            fdefs.add_field_definition(
                name, ftype, dtype_map[dtype], rank, int(nstates))
    return fdefs

class CFDMesh(StkMesh):
    """Representation of a CFD Exodus-II database

    Parameters available to control reading an input mesh and writing results database

    .. code-block:: yaml

       input_db:
         filename: nrel5mw.exo       # mandatory
         auto_decomp: yes            # optional: yes/no
         auto_decomp_type: rcb       # optional: rcb, rib, etc.
         mesh_read_type: mesh        # optional: mesh, restart
         read_all_fields: no         # optional: yes/no
         read_from: latest_time      # optional: latest_time, start_time, time_step
         start_time: 10.0            # Parameter read if `read_from: start_time`
         read_step: 200              # Parameter read if `read_from: time_step`
         time_match_option: closest  # optional: closest, linear_interpolation
         create_edges: no            # optional: yes/no

       output_db:
         filename: nrel5mw.e
         auto_join: no               # optional: yes/no
         write_type: results         # optional: results, restart
         write_time: 0.0             # optional
    """

    _dbpurpose_map = dict(
        read_mesh=stk.DatabasePurpose.READ_MESH,
        read_restart=stk.DatabasePurpose.READ_RESTART,
        write_results=stk.DatabasePurpose.WRITE_RESULTS,
        write_restart=stk.DatabasePurpose.WRITE_RESTART)
    _tmo_map = dict(
        closest=stk.TimeMatchOption.CLOSEST,
        linear_interpolation=stk.TimeMatchOption.LINEAR_INTERPOLATION)

    def __init__(self, par):
        """Initialize the CFD mesh instance"""
        super().__init__(par)
        #: Track list of input fields to be read
        self.input_fields = set()
        #: Track list of output fields registered
        self.output_fields = set()
        self.field_defs = _create_field_definitions(
            ndim=self.meta.spatial_dimension)
        self.fields = dict()

    def determine_exodusdb_state(self, filename):
        """Perform various checks and return information regarding the exodus database"""
        # If this is an auto-generated mesh, perform no checks
        if filename[:10] == "generated:":
            return True

        fpath = Path(filename)
        has_single_file = fpath.exists()
        par_file_pat = "%s.%d.*"%(filename.strip(), self.comm.size)
        has_par_file = len(glob.glob(par_file_pat)) == self.comm.size

        if not (has_single_file or has_par_file):
            raise RuntimeError("Cannot find requested input file")
        return has_par_file

    def init_mesh_meta(self, options):
        """Process the dictionary and initialize metadata if necessary"""
        filename = options.filename
        has_parallel = self.determine_exodusdb_state(filename)
        auto_decomp = options.get("auto_decomp", (not has_parallel))
        auto_decomp_type = options.get("auto_decomp_type", "rcb")
        read_all_fields = options.get("read_all_fields", False)
        read_type = "read_" + options.get("read_type", "mesh")
        tmo_inp = options.get("time_match_option", "closest")
        tmo = self._tmo_map[tmo_inp]
        if read_type not in self._dbpurpose_map:
            raise KeyError("Invalid read type provided: " + options.get("read_type"))
        purpose = self._dbpurpose_map[read_type]
        self.read_mesh_meta_data(
            filename, purpose=purpose, auto_decomp=auto_decomp,
            auto_decomp_type=auto_decomp_type, auto_declare_fields=read_all_fields,
            tmo=tmo)

    def read_input_fields(self, options):
        """Process options and read the necessary input fields"""
        time1 = 0.0
        read_all_fields = options.get("read_all_fields", False)
        if not (read_all_fields or self.input_fields):
            return time1

        meta = self.meta
        stkio = self.stkio
        for fname in self.input_fields:
            fld = meta.get_field_by_name(fname)
            if not fld.is_null:
                stkio.add_input_field(fld)
        read_time_type = options.get("read_from", "latest_time")
        if read_time_type == "time_step":
            time_step = options.read_step
            if time_step > stkio.num_time_steps:
                raise RuntimeError("Requested time step (%d) not found in database"%
                                   time_step)
            time1, missing = stkio.read_defined_input_fields_at_step(time_step)
            _lgr.info("CFDMesh: read fields from time = %f", time1)
            if missing:
                _lgr.warning("CFDMesh: missing fields during read: %s", missing)
            return time1

        read_time = (options.start_time
                     if read_time_type == "start_time"
                     else stkio.max_time)
        time1, missing = stkio.read_defined_input_fields(read_time)
        _lgr.info("CFDMesh: read fields from time = %f", time1)
        if missing:
            _lgr.warning("CFDMesh: missing fields during read: %s", missing)
        return time1

    def init_mesh_bulk(self, options):
        """Process the dictionary and initialize bulkdata"""
        create_edges = options.get("create_edges", False)
        self.populate_bulk_data(create_edges=create_edges)
        return self.read_input_fields(options)

    def write_output_db(self, options, time=0.0):
        """Write outputs"""
        filename = options.filename
        write_type = "write_" + options.get("write_type", "results")
        output_time = options.get("write_time", time)
        purpose = self._dbpurpose_map[write_type]
        auto_join = options.get("auto_join", False)
        if auto_join:
            self.set_auto_join(output=auto_join, restart=auto_join)
        stkio = self.stkio
        meta = self.meta
        _lgr.info("CFDMesh: creating output mesh = %s", filename)
        fh = stkio.create_output_mesh(filename, purpose=purpose)
        for fname in self.output_fields:
            fld = meta.get_field_by_name(fname)
            if not fld.is_null:
                stkio.add_field(fh, fld)
        _lgr.info("CFDMesh: writing fields at time = %f", output_time)
        stkio.begin_output_step(fh, output_time)
        stkio.write_defined_output_fields(fh)
        stkio.end_output_step(fh)

    def add_inout_fields(self, field_names):
        """Add list of fields to both input and output fields"""
        self.add_input_fields(field_names)
        self.add_output_fields(field_names)

    def add_input_fields(self, field_names):
        """Add to input fields"""
        self.input_fields.update(field_names)

    def add_output_fields(self, field_names):
        """Add to output fields"""
        self.output_fields.update(field_names)

    def _declare_dbl_field(self, fdef, num_states):
        """Declare a normal double field"""
        meta = self.meta
        nstates = num_states or fdef.num_states
        fld = None
        if fdef.ftype == "scalar":
            fld = meta.declare_scalar_field(fdef.name, rank=fdef.rank,
                                            number_of_states=nstates)
        elif fdef.ftype == "vector":
            fld = meta.declare_scalar_field(fdef.name, rank=fdef.rank,
                                            number_of_states=nstates)
        else:
            fld = meta.declare_generic_field(fdef.name, rank=fdef.rank,
                                             number_of_states=nstates)
        return fld

    def declare_cfd_field(
            self, field_name,
            sel=None, init_value=None, num_states=None):
        """Declare a CFD field"""
        if field_name in self.fields:
            return self.fields[field_name]

        if not field_name in self.field_defs:
            raise KeyError("Unknown CFD field requested: %s"%field_name)
        fdef = self.field_defs[field_name]
        if fdef.ftype == "generic":
            raise NotImplementedError("Generic fields not supported yet")
        fld = self._declare_dbl_field(fdef, num_states)

        num_comp = 1 if fdef.ftype == "scalar" else self.meta.spatial_dimension
        if sel is not None:
            fld.add_to_selected(sel, num_comp, init_value)

        self.fields[field_name] = fld
        return fld
