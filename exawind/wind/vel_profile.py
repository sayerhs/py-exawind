# -*- coding: utf-8 -*-

"""\
Commonly used velocity profiles

"""

import abc
import inspect
import numpy as np
from ..prelude.struct import Struct

class VelocityProfile(abc.ABC):
    """Baseclass for velocity profiles"""

    _available_profiles = dict()

    @classmethod
    def available_profiles(cls):
        """Get the names of the available profiles"""
        return cls._available_profiles.keys()

    @classmethod
    def get_profile_cls(cls, profile_name):
        """Return the class for a given profile"""
        if profile_name not in cls._available_profiles:
            raise KeyError("Unknown velocity profile type: %s"%profile_name)
        return cls._available_profiles[profile_name]

    def __init_subclass__(cls, profile_name, **kwargs):
        """Register a velocity profile type"""
        if profile_name in cls._available_profiles:
            raise KeyError("Velocity profile = %s already registered with %s"%(
                profile_name, cls._available_profiles[profile_name].__name__))
        super().__init_subclass__(**kwargs)
        cls._available_profiles[profile_name] = cls
        cls._profile_name = profile_name

    @abc.abstractmethod
    def __call__(self, heights):
        """Return the velocities for a given height"""

class ConstantProfile(VelocityProfile, profile_name="constant"):
    """Constant velocity profile"""

    def __init__(self, ref_speed):
        """
        Args:
            ref_speed (double): Reference speed
        """
        self.ref_speed = ref_speed

    def __call__(self, heights):
        """Return computed velocities"""
        return np.ones_like(heights) * self.ref_speed

class LogLawProfile(VelocityProfile, profile_name="log_law"):
    """A logarithmic velocity profile"""

    def __init__(self, ref_speed, ref_height, roughness_height, kappa=0.41):
        """
        Args:
            uref (double): Reference speed
            zref (double): Height at which reference speed is provided
            z0 (double): roughness height
        """
        # pylint: disable=invalid-name
        self.uref = ref_speed
        self.zref = ref_height
        self.z0 = roughness_height
        self.kappa = kappa

        self.ustar = kappa * ref_speed / np.log(ref_height / roughness_height)

    def __call__(self, heights):
        """Compute log profile at given heights"""
        z0 = self.z0
        htmp = np.asarray(heights)
        ht = np.where(htmp > z0, htmp, z0)
        return (self.ustar / self.kappa) * np.log(ht / z0)

class PowerLawProfile(VelocityProfile, profile_name="power_law"):
    """A power law velocity profile"""

    def __init__(self, ref_speed, ref_height, shear_exponent,
                 offset_height=0.0, min_speed=0.0, max_speed=1.0e8):
        """
        Args:
            uref (double): Reference speed (m/s)
            zref (double): Reference height (m)
            alpha (double): Shear exponent
            zoffset (double): Offset for shifting height
            min_speed (double): Minimum cutoff velocity
            max_speed (double): Maximum cutoff velocity
        """
        self.uref = ref_speed
        self.zref = ref_height
        self.alpha = shear_exponent
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.zoffset = offset_height

    def __call__(self, heights):
        """Compute power law profile at given heights

        Args:
            heights: Array-like object of heights where profile is evaluated
        """
        htmp = np.asarray(heights) - self.zoffset
        ht = np.where(htmp > 0, htmp, 0.0)
        utmp = self.uref * np.power((ht/self.zref), self.alpha)
        utmp1 = np.where(utmp > self.min_speed, utmp, self.min_speed)
        utmp2 = np.where(utmp1 < self.max_speed, utmp1, self.max_speed)
        return utmp2

def velocity_profile(opts, key="profile_type"):
    """Get the velocity profile instance"""
    profile_type = opts[key]
    prof_cls = VelocityProfile.get_profile_cls(profile_type)
    sig = inspect.signature(prof_cls.__init__)
    params = sig.parameters
    req_args = []
    opt_args = []
    # Skip the first argument (always `self`)
    arg_names = list(params.keys())[1:]
    for key in arg_names:
        arg = params[key]
        if arg.default == inspect.Parameter.empty:
            req_args.append(key)
        else:
            opt_args.append(key)
    init_args = dict()
    for arg in req_args:
        init_args[arg] = opts[arg]

    for arg in opt_args:
        if arg in opts:
            init_args[arg] = opts[arg]

    return prof_cls(**init_args)
