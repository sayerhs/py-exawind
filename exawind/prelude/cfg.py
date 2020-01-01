# -*- coding: utf-8 -*-

"""\
Configuration manager
~~~~~~~~~~~~~~~~~~~~~
"""

import os
import abc
import inspect
import logging
from logging.config import dictConfig
from pathlib import Path

class ConfigManager(metaclass=abc.ABCMeta):
    """Base configuration manager utility"""

    def __init__(self):
        self.cfg = None

    @abc.abstractstaticmethod
    def rc_type():
        """Type of configuration file"""

    @abc.abstractstaticmethod
    def rc_base():
        """Base filename"""

    @abc.abstractstaticmethod
    def cfg_class():
        """Configuration class"""

    @property
    def cfg_root(self):
        """Root node of the configuration"""
        return self.rc_base()

    @property
    def rc_envvar(self):
        """Environment variable for searching RC files"""
        return "%sRC"%self.rc_base()

    @property
    def rc_sys_envvar(self):
        """Environment variable for searching system RC files"""
        return "%sRC_SYSTEM"%self.rc_base()

    @property
    def rc_file_ext(self):
        """File extension for configuration file"""
        return self.rc_type()

    @property
    def rc_home(self):
        """Home config file"""
        return "." + self.rc_base() + "rc"

    @property
    def cfg_file_name(self):
        """Configuration file name"""
        return self.rc_base() + "." + self.rc_file_ext

    @property
    def cfg_files(self):
        """Return a list of available config files available on the system"""
        rcfiles = []

        sys_rc = os.environ.get(self.rc_sys_envvar, None)
        if sys_rc and Path(sys_rc).exists():
            rcfiles.append(Path(sys_rc))

        home_rc = Path.home() / self.rc_home
        if home_rc.exists():
            rcfiles.append(home_rc)

        env_rc = os.environ.get(self.rc_envvar, None)
        if env_rc and Path(env_rc).exists():
            rcfiles.append(Path(env_rc))

        cwd_rc = Path.cwd() / self.cfg_file_name
        if cwd_rc.exists():
            rcfiles.append(cwd_rc)

        return rcfiles

    @property
    def default_cfg_file(self):
        """Get default configuration file"""
        try:
            cfile = inspect.getfile(self.__class__)
            cdir = Path(cfile).parent
            default_yaml = cdir / self.cfg_file_name
            return default_yaml
        except TypeError:
            return self.cfg_file_name

    @property
    def default_config(self):
        """Return default config"""
        cfg_cls = self.cfg_class()
        cfg_file = Path(self.default_cfg_file)
        if not cfg_file.exists():
            return cfg_cls()

        return self.load_cfg_file(cfg_file)

    def load_cfg_file(self, cfg_file):
        """Load a configuration file"""
        cfg_cls = self.cfg_class()
        cfg = cfg_cls.load_file(cfg_file)
        return cfg

    def reset_to_defaults(self):
        """Reset to default configuration"""
        self.cfg = self.default_config
        return self.cfg

    @staticmethod
    def configure_logging(log_cfg=None):
        """Configure python logging"""
        if log_cfg is None:
            logging.basicConfig()
        else:
            logger_cfg = log_cfg.pylogger_options
            dictConfig(logger_cfg)

    def init_config(self, base_cfg=None, init_logging=True):
        """Initialize configuration"""
        cfg = base_cfg or self.default_config
        rcfiles = self.cfg_files
        for rcname in rcfiles:
            cfg.merge(self.load_cfg_file(rcname))

        if init_logging:
            cfg_root = cfg.get(self.cfg_root, self.cfg_class()())
            log_cfg = cfg_root.get("logging", None)
            self.configure_logging(log_cfg)
        self.cfg = cfg
        return cfg

    def get_config(self, base_cfg=None, init_logging=True):
        """Get the current configuration object"""
        if self.cfg is None:
            self.init_config(base_cfg, init_logging)
        return self.cfg

def make_config_manager(cls):
    """Make a configuration object"""
    cfg_obj = cls()
    def config_manager():
        """Configuration manager"""
        return cfg_obj
    return config_manager
