# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""\
Mesh and field transformations
------------------------------
"""

import logging
import numpy as np
import stk
from ...prelude.struct import Struct
from ...prelude.euclid import TrMat, Vec
from ..task import NaluTask
from .. import mesh_ops

_lgr = logging.getLogger(__name__)

class TransformMesh(NaluTask, task_name="transform_mesh"):
    """Perform transformations on mesh and its fields

    Supported transformations are rotation and translation
    """
    # pylint: disable=arguments-differ

    def __init__(self, cfd_mesh):
        """
        Args:
            cfd_mesh (CFDMesh): The mesh instance
        """
        super().__init__(cfd_mesh)
        #: Transformations list
        self.transforms = None
        #: Default parts
        self.part_names = None
        #: Field names for rotation
        self.field_names = None
        #: List of field instances
        self.fields = None

    @staticmethod
    def _process_rotation(options):
        """Process a rotation definition"""
        rot_map = dict(quarternion=TrMat.Q, rot_x=TrMat.X,
                       rot_y=TrMat.Y, rot_z=TrMat.Z, axes=TrMat.axes)
        rot_type = options.get("rotation_type", "quarternion")
        if rot_type not in rot_map:
            raise KeyError("Unknown rotation type: %s"%rot_type)
        origin = Vec(options.get("origin", Vec.zero()))
        tmat = None
        if rot_type == "quarternion":
            tmat = TrMat.Q(axis=np.asarray(options.axis), angle=options.angle)
        elif rot_type[:4] == "rot_":
            tmat = rot_map[rot_type](angle=options.angle)
        else:
            tmat = rot_map[rot_type](x=options.xdir, y=options.ydir, z=options.zdir)

        parts = options.get("parts", None)
        return Struct(ttype="rotation", parts=parts, rot_mat=tmat, origin=origin)

    @staticmethod
    def _process_translation(options):
        """Process translation definition"""
        trvec = Vec(options.offset_vector)
        parts = options.get("parts", None)
        return Struct(ttype="translation", parts=parts, trans_vec=trvec)

    def process_inputs(self, options, default_parts=None):
        """Parse inputs"""
        def process_transforms(transforms):
            """Helper method to process transforms"""
            for tt in transforms:
                for ttype, val in tt.items():
                    func = getattr(self, "_process_%s"%ttype, None)
                    assert func is not None, "Invalid transformation type: %s"%ttype
                    yield func(val)

        _lgr.info("transform_mesh: processing inputs")
        opts = Struct(**options)
        assert "transforms" in opts, "transform_mesh requires transforms"
        self.part_names = opts.get("parts", default_parts)
        self.transforms = list(process_transforms(opts.transforms))

        # Handle field rotations
        self.field_names = opts.get("fields", None)

    def init_meta_data(self):
        """Initialize the meta data"""
        _lgr.info("transform_mesh: initializing meta data")
        default_parts = None
        need_parts = True
        if self.part_names is not None:
            default_parts = list(self.names_to_parts(self.part_names))
            need_parts = False

        for tt in self.transforms:
            pnames = tt.parts
            if need_parts and (pnames is None):
                raise ValueError("No parts found for transformation")
            tt.parts = self.names_to_parts(pnames) if pnames else default_parts

        if self.field_names:
            meta = self.mesh.meta
            self.fields = [meta.get_field_by_name(ff, must_exist=True)
                           for ff in self.field_names]
            self.mesh.add_inout_fields(self.field_names)
        _lgr.info("transform_mesh: meta data initialization completed")

    def transform_mesh(self):
        """Perform mesh transformations"""
        _lgr.info("transform_mesh: begin mesh transformations")
        meta = self.mesh.meta
        for tt in self.transforms:
            sel = (stk.StkSelector.select_union(tt.parts)
                   & (meta.locally_owned_part | meta.globally_shared_part))
            if tt.ttype == "rotation":
                mesh_ops.rotate_coordinates(
                    mesh=self.mesh, trmat=tt.rot_mat, origin=tt.origin, sel=sel)
            elif tt.ttype == "translation":
                mesh_ops.translate_mesh(self.mesh, tt.trans_vec, sel=sel)
        _lgr.info("transform_mesh: mesh transformation completed")

    def transform_fields(self):
        """Perform field transformations"""
        _lgr.info("transform_mesh: begin field transformations")
        meta = self.mesh.meta
        for fld in self.fields:
            for tt in self.transforms:
                if tt.ttype != "rotation":
                    continue
                sel = (stk.StkSelector.select_union(tt.parts)
                       & (meta.locally_owned_part | meta.globally_shared_part))
                mesh_ops.rotate_field(self.mesh, fld, tt.rot_mat, sel=sel,
                                      rank=fld.entity_rank)
        _lgr.info("transform_mesh: field transformation completed")

    def run(self):
        """Perform the transformations requested by user"""
        _lgr.info("transform_mesh: begin execution")
        self.transform_mesh()
        if self.fields:
            self.transform_fields()
        _lgr.info("transform_mesh: end execution")
        mesh_modified = True
        return mesh_modified
