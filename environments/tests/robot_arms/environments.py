#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:58:44 2022

@author: nnavarroguerrero
"""

from environments.robot_arms import kuka_lbr_iiwa as kuka

print("NEW TEST")
arm = kuka.KUKA_LBR_IIWA()
arm.create_dataset(10, "training.csv")
