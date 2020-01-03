# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""\
Initial fields for ABL simulations
----------------------------------
"""

import logging
import numpy as np
import stk
from ...prelude.struct import Struct
from ...prelude.euclid import TrMat, Vec
from ...wind.vel_profile import VelocityProfile, velocity_profile
from ..task import NaluTask
from .. import mesh_ops

_lgr = logging.getLogger(__name__)

class WindFields(NaluTask, task_name="wind_fields"):
    """Initialize CFD fields for wind applications"""

    def __init__(self, mesh):
        """
        Args:
            mesh (CFDMesh): The mesh instance
        """
        super().__init__(mesh)
        #: User defined options dictionary
        self.options = None
        #: List of part names where the fields are generated
        self.part_names = None
        #: List of parts
        self.parts = []
        #: Velocity information
        self.vel_info = Struct()
        #: Temperature information
        self.temp_info = Struct()

    def process_inputs(self, options, default_parts=None):
        """Parse inputs"""
        _lgr.info("abl_fields: processing inputs")
        self.options = options
        self.part_names = options.get("parts", default_parts)
        if self.part_names is None:
            _lgr.warning("No part names found, generating fields for all parts in mesh")

        if "velocity" in options:
            self.vel_info.do_velocity = True
            self._process_velocity_inputs(options.velocity)
        else:
            self.vel_info.do_velocity = False

        if "temperature" in options:
            self.temp_info.do_temperature = True
            self._process_temperature_inputs(options.temperature)
        else:
            self.temp_info.do_temperature = False

    def init_meta_data(self):
        """Initialize parts"""
        meta = self.mesh.meta
        if self.part_names:
            self.parts = list(self.names_to_parts(self.part_names))
        else:
            self.parts = [meta.universal_part]
        sel = stk.StkSelector.select_union(self.parts)
        if self.vel_info.do_velocity:
            self.mesh.declare_cfd_field("velocity", sel=sel, num_states=1)
            self.mesh.add_output_fields(["velocity"])
        if self.temp_info.do_temperature:
            self.mesh.declare_cfd_field("temperature", sel=sel)
            self.mesh.add_output_fields(["temperature"])
            tinfo = self.temp_info
            if "perturbations" in tinfo and tinfo.periodic_names:
                tinfo.periodic_parts = list(self.names_to_parts(tinfo.periodic_names))

    def _process_velocity_inputs(self, opts):
        """Process the velocity input node"""
        if "perturbations" in opts:
            popts = opts["perturbations"]
            vinfo = self.vel_info
            vinfo.perturbations = Struct()
            vinfo.ref_height = popts["reference_height"]
            vinfo.amplitude = popts["amplitude"]
            vinfo.periods = popts["periods"]

        vel_type = opts.get("type", "table_lookup")
        if vel_type == "table_lookup":
            self._process_velocity_table_lookup(opts)
            return

        # Check that `vel_type` is one of the registered profiles
        assert vel_type in VelocityProfile.available_profiles(), (
            "Unknown velocity profile type: %s"%vel_type)
        self._process_velocity_profile_def(opts)

    def _process_temperature_inputs(self, opts):
        """Process the temperature input node"""
        temp_type = opts.get("type", "table_lookup")
        assert temp_type == "table_lookup", "Only table lookup supported for temperature"
        heights = np.asarray(opts.heights)
        temperature = np.asarray(opts["values"])
        assert heights.ndim == 1, "Invalid array provided for heights"
        assert temperature.shape == heights.shape, "height, temperature values inconsistent"
        tinfo = self.temp_info
        tinfo.heights = heights
        tinfo.temperature = temperature

        if "perturbations" in opts:
            popts = opts["perturbations"]
            tinfo.perturbations = Struct()
            tinfo.cutoff_height = popts["cutoff_height"]
            tinfo.amplitude = popts["amplitude"]
            tinfo.gauss_mean = popts.get("random_gauss_mean", 0.0)
            tinfo.gauss_var = popts.get("random_gauss_var", 1.0)
            tinfo.periodic_names = popts.get("skip_periodic_parts", [])
            tinfo.periodic_parts = []

    def _process_velocity_table_lookup(self, opts):
        """Table lookup information"""
        heights = np.asarray(opts.heights)
        velocities = np.asarray(opts["values"])
        assert heights.ndim == 1, "Invalid array provided for heights"
        assert ((velocities.ndim == 2) and velocities.shape == (len(heights), 3))
        vinfo = self.vel_info
        vinfo.heights = heights
        vinfo.velocities = velocities
        vinfo.init_func = self.init_velocity_table_lookup

    def init_velocity_table_lookup(self):
        """Initialize velocities using table lookup"""
        # pylint: disable=import-outside-toplevel
        _lgr.info("wind_fields: initializing velocity using table lookup")
        import scipy.interpolate as sint
        meta = self.mesh.meta
        vinfo = self.vel_info
        velinterp = sint.interp1d(vinfo.heights, vinfo.velocities.T)
        sel = (stk.StkSelector.select_union(self.parts) &
               (meta.locally_owned_part | meta.globally_shared_part))

        coords = meta.coordinate_field
        vfield = meta.get_field("velocity", stk.StkRank.NODE_RANK)
        for bkt in self.mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            vel = vfield.bkt_view(bkt)
            vtmp = velinterp(xyz[:, 2])
            vel[:, :] = vtmp.T
        _lgr.info("wind_fields: velocity initialized using table lookup")

    def _process_velocity_profile_def(self, opts):
        """Process pre-defined velocity profile information"""
        vinfo = self.vel_info
        vinfo.profile_type = opts["type"]
        vinfo.vel_func = velocity_profile(opts, "type")
        if "orientation" in opts:
            orient_opts = opts["orientation"]
            otype = orient_opts.get("type", "wind_direction")
            vinfo.orientation = otype
            if otype == "wind_direction":
                wdir = orient_opts["wind_direction"]
                assert (0.0 <= wdir <= 360.0), (
                    "Wind direction must be in terms of compass angle")
                angle = 90.0 - wdir
                vinfo.tmat = TrMat.Z(angle)
            else:
                vinfo.tmat = TrMat.from_dict(orient_opts)
            vinfo.origin = orient_opts.get("origin", Vec.zero())
        else:
            vinfo.orientation = "constant"
            vinfo.origin = Vec.zero()
            vinfo.tmat = TrMat.I()
        vinfo.init_func = self.init_vel_predef_profile

    def init_vel_predef_profile(self):
        """Initialize a predefined velocity profile"""
        vinfo = self.vel_info
        if vinfo.orientation == "wind_direction":
            # Avoid some computations for wind-direction based initialization
            self.init_vel_profile_wdir()
        else:
            self.init_vel_prof_transform()

    def init_vel_profile_wdir(self):
        """Initialize a predefined velocity profile with wind direction"""
        vinfo = self.vel_info
        _lgr.info("wind_fields: initializing %s profile", vinfo.profile_type)
        meta = self.mesh.meta
        sel = (stk.StkSelector.select_union(self.parts) &
               (meta.locally_owned_part | meta.globally_shared_part))

        coords = meta.coordinate_field
        vfield = meta.get_field("velocity", stk.StkRank.NODE_RANK)
        vfunc = vinfo.vel_func
        trmat = vinfo.tmat
        origin = vinfo.origin
        for bkt in self.mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            vel = vfield.bkt_view(bkt)
            vprof = vfunc(xyz[:, -1] - origin[-1])
            vlocal = np.zeros_like(xyz).view(Vec)
            vlocal[:, 0] = vprof
            vel[:, :] = - np.asarray(vlocal * trmat)
        _lgr.info("wind_fields: initialized %s profile", vinfo.profile_type)

    def init_vel_prof_transform(self):
        """Initialize pre-defined velocity profile with axis transformations"""
        vinfo = self.vel_info
        _lgr.info("wind_fields: initializing %s profile", vinfo.profile_type)
        meta = self.mesh.meta
        sel = (stk.StkSelector.select_union(self.parts) &
               (meta.locally_owned_part | meta.globally_shared_part))

        coords = meta.coordinate_field
        vfield = meta.get_field("velocity", stk.StkRank.NODE_RANK)
        vfunc = vinfo.vel_func
        trmat = vinfo.tmat
        origin = vinfo.origin
        for bkt in self.mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            vel = vfield.bkt_view(bkt)
            xyz_t = trmat * Vec(xyz - origin)
            vprof = vfunc(xyz_t[:, -1])
            vlocal = np.zeros_like(xyz).view(Vec)
            vlocal[:, 0] = vprof
            vel[:, :] = np.asarray(vlocal * trmat)
        _lgr.info("wind_fields: initialized %s profile", vinfo.profile_type)

    def perturb_velocity_field(self):
        """Add sinusoidal perturbations to velocity field"""
        vinfo = self.vel_info
        _lgr.info("wind_fields: adding sinusoidal fluctuations to velocity")
        meta = self.mesh.meta
        sel = (stk.StkSelector.select_union(self.parts) &
               (meta.locally_owned_part | meta.globally_shared_part))
        coords = meta.coordinate_field
        vfield = meta.get_field("velocity")

        box_min, box_max = mesh_ops.bbox(self.mesh, sel=sel)
        box_len = (box_max - box_min)
        periods = np.asarray(vinfo.periods)
        amplitude = np.asarray(vinfo.amplitude)
        aval = 2.0 * np.pi * periods[0] / box_len[1]
        bval = 2.0 * np.pi * periods[1] / box_len[0]
        vel_fac = amplitude * np.exp(0.5) / vinfo.ref_height

        for bkt in self.mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            vel = vfield.bkt_view(bkt)
            xl = xyz[:, 0] - box_min[0]
            yl = xyz[:, 1] - box_min[1]
            zl = xyz[:, 2] / vinfo.ref_height
            damp = np.exp(-0.5 * zl * zl)
            vel[:, 0] += vel_fac[0] * damp * xyz[:, 2] * np.cos(aval * yl)
            vel[:, 1] += vel_fac[1] * damp * xyz[:, 2] * np.sin(bval * xl)
        _lgr.info("wind_fields: done adding sinusoidal fluctuations to velocity")

    def init_temperature_table_lookup(self):
        """Initialize velocities using table lookup"""
        # pylint: disable=import-outside-toplevel
        _lgr.info("wind_fields: initializing temperature using table lookup")
        import scipy.interpolate as sint
        meta = self.mesh.meta
        tinfo = self.temp_info
        tinterp = sint.interp1d(tinfo.heights, tinfo.temperature)
        sel = (stk.StkSelector.select_union(self.parts) &
               (meta.locally_owned_part | meta.globally_shared_part))

        coords = meta.coordinate_field
        tfield = meta.get_field("temperature", stk.StkRank.NODE_RANK)
        for bkt in self.mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            temp_arr = tfield.bkt_view(bkt)
            temp_vals = tinterp(xyz[:, 2])
            temp_arr[:] = temp_vals
        _lgr.info("wind_fields: temperature initialized using table lookup")

    def perturb_temperature_field(self):
        """Add perturbations to temperature field"""
        _lgr.info("wind_fields: adding sinusoidal fluctuations to temperature")
        tinfo = self.temp_info
        meta = self.mesh.meta
        sel = (stk.StkSelector.select_union(self.parts) &
               stk.StkSelector.select_union(tinfo.periodic_parts).complement() &
               (meta.locally_owned_part | meta.globally_shared_part))
        coords = meta.coordinate_field
        tfield = meta.get_field("temperature")
        amplitude = tinfo.amplitude
        gauss_mean = tinfo.gauss_mean
        gauss_var = tinfo.gauss_var
        cutoff_height = tinfo.cutoff_height
        for bkt in self.mesh.iter_buckets(sel):
            xyz = coords.bkt_view(bkt)
            temp_arr = tfield.bkt_view(bkt)
            perturb = np.random.normal(gauss_mean, gauss_var, bkt.size)
            temp_arr[:] += amplitude * np.where(xyz[:, -1] < cutoff_height, perturb, 0.0)

    def init_velocity_field(self):
        """Process velocity field"""
        self.vel_info.init_func()
        if "perturbations" in self.vel_info:
            self.perturb_velocity_field()

    def init_temperature_field(self):
        """Process the temperature field"""
        self.init_temperature_table_lookup()
        if "perturbations" in self.temp_info:
            self.perturb_temperature_field()

    def run(self):
        """Initialize the fields"""
        _lgr.info("wind_fields: begin execution")
        if self.vel_info.do_velocity:
            self.init_velocity_field()
        if self.temp_info.do_temperature:
            self.init_temperature_field()
        _lgr.info("wind_fields: end execution")
        mesh_modified = True
        return mesh_modified
