# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

"""
vaayu.prelude.struct Tests
"""

import numpy as np
from numpy.testing import assert_allclose
from exawind.prelude.struct import Struct

def test_struct_access():
    dmap = Struct(
        a=1, b=2,
        c=Struct(x=10, y=20, z=[10, 20, 2.0e-5]))
    assert dmap.b == dmap["b"]
    assert dmap.c.x == 10
    assert len(dmap["c"].z) == 3
    assert_allclose(dmap.c.z[-1], 2.0e-5)

yaml_sample = """
nalu_preprocess:
  # Name of the input exodus database
  input_db: ekmanSpiral.g
  # Name of the output exodus database
  output_db: ekmesh.g
  # Flag indicating whether the database contains 8-bit integers
  ioss_8bit_ints: false

  # Flag indicating mesh decomposition type
  # automatic_decomposition_type: rcb

  # Nalu preprocessor expects a list of tasks to be performed on the mesh and
  # field data structures
  tasks:
    - init_abl_fields
    - generate_planes

  # Inputs for each "task" is organized under the section corresponding to the
  # task name
  init_abl_fields:
    fluid_parts: [Unspecified-2-HEX]

    temperature:
      heights: [    0, 650.0, 750.0, 10750.0]
      values:  [280.0, 280.0, 288.0,   318.0]

    velocity:
      heights: [0.0, 10.0, 30.0, 70.0, 100.0, 650.0, 10000.0]
      values:
        - [ 0.0, 0.0, 0.0]
        - [4.81947, -4.81947, 0.0]
        - [5.63845, -5.63845, 0.0]
        - [6.36396, -6.36396, 0.0]
        - [6.69663, -6.69663, 0.0]
        - [8.74957, -8.74957, 0.0]
        - [8.74957, -8.74957, 0.0]

"""

json_sample = '{"nalu_preprocess": {"input_db": "ekmanSpiral.g", "output_db": "ekmesh.g", "ioss_8bit_ints": false, "tasks": ["init_abl_fields", "generate_planes"], "init_abl_fields": {"fluid_parts": ["Unspecified-2-HEX"], "temperature": {"heights": [0, 650.0, 750.0, 10750.0], "values": [280.0, 280.0, 288.0, 318.0]}}}}'

def test_struct_yaml_input():
    dmap = Struct.from_yaml(yaml_sample)
    tmp = dmap.nalu_preprocess
    assert len(tmp.tasks) == 2
    assert tmp.init_abl_fields.fluid_parts[0] == "Unspecified-2-HEX"

    temp_values = np.array([280.0, 280.0, 288.0, 318.0] )
    temp_data = np.array(tmp.init_abl_fields.temperature["values"])
    assert_allclose(temp_data, temp_values)

def test_struct_yaml_dump():
    dmap = Struct.from_yaml(yaml_sample)
    txt = dmap.to_yaml(default_flow_style=True)
    assert txt

def test_struct_json_input():
    dmap = Struct.from_json(json_sample)
    tmp = dmap.nalu_preprocess
    assert len(tmp.tasks) == 2
    assert tmp.init_abl_fields.fluid_parts[0] == "Unspecified-2-HEX"

    temp_values = np.array([280.0, 280.0, 288.0, 318.0] )
    temp_data = np.array(tmp.init_abl_fields.temperature["values"])
    assert_allclose(temp_data, temp_values)

def test_struct_json_dump():
    dmap = Struct.from_json(json_sample)
    txt = dmap.to_json(indent=None)
    assert txt
