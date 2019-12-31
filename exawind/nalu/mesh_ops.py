# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""\
Mesh and field manipulation utilities
-------------------------------------

"""

import operator
import numpy as np
from mpi4py import MPI
import stk
from ..prelude.euclid import Vec

def field_minmax(mesh, field, ndim, sel=None, rank=stk.StkRank.NODE_RANK):
    """Return the min/max range for a given field

    Args:
        mesh (StkMesh): A STK Mesh instance
        field (StkFieldBase): The field for which ranges are sought
        sel (StkSelector): Selector for entities to loop over
        rank (StkRank): The entity rank where the field is defined

    Return:
       (array, array): Array of (min, max) values
    """
    meta = mesh.meta
    ssel = sel or stk.StkSelector.from_part(meta.locally_owned_part)

    minval = 1.0e15 * np.ones((ndim,))
    maxval = -1.0e15 * np.ones((ndim,))
    for bkt in mesh.iter_buckets(ssel, rank):
        fld = field.bkt_view(bkt)
        minval = np.fmin(minval, np.min(fld, axis=0))
        maxval = np.fmax(maxval, np.max(fld, axis=0))

    comm = MPI.COMM_WORLD
    gminval = np.zeros_like(minval)
    gmaxval = np.zeros_like(maxval)
    comm.Allreduce(minval, gminval, op=MPI.MIN)
    comm.Allreduce(maxval, gmaxval, op=MPI.MAX)

    return (gminval, gmaxval)

def bbox(mesh, sel=None, field_name="coordinates", rank=stk.StkRank.NODE_RANK):
    """Compute the bounding box of the coordinates field

    Args:
        mesh (StkMesh): A STK mesh instance
        sel (StkSelector): A STK selector instance
        field_name (str): "coordinates" or "current_coordinates"

    Return:
       (array, array): Coordinates of the min/max corners
    """
    meta = mesh.meta
    ndim = meta.spatial_dimension
    coords = meta.get_field(field_name, must_exist=True)

    return field_minmax(mesh, coords, ndim, sel, rank)

def rotate_coordinates(mesh, trmat, origin=Vec.zero(), sel=None):
    """Rotate the coordinate field of the mesh

    Args:
        mesh (StkMesh): The mesh instance
        trmat (TrMat): Transformation matrix
        origin (Vec): Origin about which the rotation is performed
        sel (StkSelector): Selector for entities where field rotation is applied
    """
    meta = mesh.meta
    coords = meta.coordinate_field
    ssel = sel or (meta.locally_owned_part | meta.globally_shared_part)

    for bkt in mesh.iter_buckets(ssel, stk.StkRank.NODE_RANK):
        xyz = coords.bkt_view(bkt)
        mxyz = Vec(xyz) - origin
        new_xyz = mxyz * trmat
        xyz[:, :] = np.asarray(new_xyz + origin)[:, :]

def rotate_current_coordinates(
        mesh, trmat,
        origin=Vec.zero(), sel=None,
        field_name="current_coordinates"):
    """Rotate current coordinates based on model coordinates

    Args:
        mesh (StkMesh): The mesh instance
        trmat (TrMat): Transformation matrix
        origin (Vec): Origin about which the rotation is performed
        sel (StkSelector): Selector for entities where field rotation is applied
        field_name (str): Name of the current coordinates field
    """
    meta = mesh.meta
    coords = meta.coordinate_field
    curr_coords = meta.get_field(field_name, must_exist=True)
    ssel = sel or (meta.locally_owned_part | meta.globally_shared_part)

    for bkt in mesh.iter_buckets(ssel, stk.StkRank.NODE_RANK):
        xyz = coords.bkt_view(bkt)
        cxyz = curr_coords.bkt_view(bkt)
        mxyz = Vec(xyz) - origin
        new_xyz = mxyz * trmat
        cxyz[:, :] = np.asarray(new_xyz + origin)

def rotate_field(mesh, field, trmat, sel=None, rank=stk.StkRank.NODE_RANK):
    """Rotate a vector field

    Args:
        mesh (StkMesh): The mesh instance
        fields (StkFieldBase): List of fields to rotate
        trmat (TrMat): Transformation matrix
        sel (StkSelector): Selector for entities where field rotation is applied
        rank (StkRank): Entity rank where the field is defined
    """
    sfield = (field
              if isinstance(field, stk.api.mesh.field.StkFieldBase)
              else mesh.meta.get_field(field, rank, must_exist=True))
    ssel = sel or stk.StkSelector.select_field(sfield)
    for bkt in mesh.iter_buckets(ssel, rank):
        fld = sfield.bkt_view(bkt)
        rot_fld = Vec(fld) * trmat
        fld[:, :] = np.asarray(rot_fld)

def rotate_fields(mesh, fields, trmat, sel=None, rank=stk.StkRank.NODE_RANK):
    """Rotate multiple fields

    Args:
        mesh (StkMesh): The mesh instance
        fields (list): List of fields to rotate
        trmat (TrMat): Transformation matrix
        sel (StkSelector): Selector for entities where field rotation is applied
        rank (StkRank): Entity rank where the field is defined
    """
    meta = mesh.meta
    ssel = sel or (meta.locally_owned_part | meta.globally_shared_part)

    field_list = [
        ff if isinstance(ff, stk.api.mesh.field.StkFieldBase)
        else meta.get_field(ff, rank, must_exist=True)
        for ff in fields]
    for bkt in mesh.iter_buckets(ssel, rank):
        for ff in field_list:
            fld = ff.bkt_view(bkt)
            rot_fld = Vec(fld) * trmat
            fld[:, :] = np.asarray(rot_fld)

def field_op(mesh, field, arg, op=operator.add, sel=None, rank=stk.StkRank.NODE_RANK):
    """Perform a field update operation

    .. code-block:: python

       import operator
       # Translate a mesh block
       field_op(mesh, coordinates, np.array([10.0, 5.0, 1.0]), op=operator.add)

       # Square the density field
       field_op(mesh, density, 2.0, op=operator.pow)

       # Multiply the velocity field by 3.0
       field_op(mesh, velocity, 3.0, op=operator.mul)

       # Subtract mean velocity
       mean_vel = np.array([10.0, 5.0, 0.0])
       field_op(mesh, velocity, mean_vel, op=operator.sub)

    Args:
        mesh (StkMesh): A STK Mesh instance
        field (StkFieldBase): The field for which ranges are sought
        sel (StkSelector): Selector for entities to loop over
        rank (StkRank): The entity rank where the field is defined
    """
    ssel = sel or stk.StkSelector.select_field(field)

    for bkt in mesh.iter_buckets(ssel, rank):
        fld = field.bkt_view(bkt)
        fld[...] = op(fld, arg)

def translate_mesh(mesh, translation_vector, sel=None, coords_name="coordinates"):
    """Translate/move the mesh

    Args:
        mesh (StkMesh): A STK Mesh instance
        translation_vector (Vec): Offset vector from current position
        sel (StkSelector): Parts over which operation is performed
        coords_name (str): Name of the coordinates field
    """
    meta = mesh.meta
    coords = meta.get_field(coords_name, must_exist=True)
    field_op(mesh, coords, translation_vector,
             op=operator.add, sel=sel, rank=stk.StkRank.NODE_RANK)
