#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:34:08 2020

@author: wxl
"""

import unittest

import numpy as np
import scipy.sparse as sp

from discretize import TensorMesh
from SimPEG import (
    maps,
    regularization,
    utils,
)


np.random.seed(10)
decimal_digit = 8 


class CrossGradient(unittest.TestCase):
    
    def setUp(self):
        
        dh = 1.0
        nx = 12
        ny = 12
        
        hx = [(dh, nx)]
        hy = [(dh, ny)]
        mesh = TensorMesh([hx, hy], "CN")
        
        # get index of center point
        midx = int(mesh.nCx/2)
        midy = int(mesh.nCy/2)
        
        # create a model1
        m1 = np.zeros((mesh.nCx, mesh.nCy))
        m1[(midx-2):(midx+2), (midy-2):(midy+2)] = 1
        
        # create a model2
        m2 = np.zeros((mesh.nCx, mesh.nCy))
        m2[(midx-3):(midx+1), (midy-3):(midy+1)] = 1
        
        m1 = utils.mkvc(m1)
        m2 = utils.mkvc(m2)
        
        # stack the true models
        models = np.r_[m1, m2]
        
        # reg
        actv = np.ones(len(m1), dtype=bool)
        plotting_map = maps.InjectActiveCells(mesh, actv, np.nan)    
        
        # maps
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))
        
        cros_grad = regularization.CrossGradient(
            mesh, 
            indActive=plotting_map.indActive,
            mapping=(wires.m1+wires.m2)
            )
        

        self._m1, self._m2 = m1, m2
        self._cros_grad = cros_grad
        self._models = models
        self._delta_m = 1e-6 # small purterbation
        self._delta_per = 1e-6 + 1 # 1.0006%

        

    def test_Jvec(self):
        """
        
        Test Jacobian matrix of cross-gradient

        """
        
        # cross-gradient values of true models
        cros_grad_0 = self._cros_grad.__call__(self._models)

        # analytic solution
        Jvec_analytic = self._cros_grad.deriv(self._models)
        
        # numerical solution
        dm1 = []
        dm2 = []
        for i in range(len(self._m1)):
            
            # dm1
            m1_tmp = self._m1.copy()
            if self._m1[i] == 0:
                m1_tmp[i] = self._delta_m
                cros_grad_1 = self._cros_grad.__call__(np.r_[m1_tmp, self._m2])
                dm1 += [(cros_grad_1 - cros_grad_0) / self._delta_m]
                
            else:
                m1_tmp[i] = m1_tmp[i] * self._delta_per
                cros_grad_1 = self._cros_grad.__call__(np.r_[m1_tmp, self._m2])
                dm1+=[(cros_grad_1 - cros_grad_0) / self._delta_m]
                
            # dm2
            m2_tmp = self._m2.copy()
            if self._m2[i] == 0:
                m2_tmp[i] = self._delta_m
                cros_grad_1 = self._cros_grad.__call__(np.r_[self._m1, m2_tmp])
                dm2 += [(cros_grad_1 - cros_grad_0) / self._delta_m]
                
            else:
                m2_tmp[i] = m2_tmp[i] * self._delta_per
                cros_grad_1 = self._cros_grad.__call__(np.r_[self._m1, m2_tmp])
                dm2+=[(cros_grad_1 - cros_grad_0) / self._delta_m]
            
            
        Jvec_numerical = np.r_[dm1, dm2]
        
        
        self.assertAlmostEqual(
            np.all(Jvec_analytic), 
            np.all(Jvec_numerical), 
            places=decimal_digit
            )
        
        
        
        
    def test_H(self):
        """
        
        Test Hessian matrix of cross-gradient

        """
        
        n = len(self._m1)
        
        # Jacobian values of true models
        J0 = self._cros_grad.deriv(self._models)
        
        # analytic solution
        H_analytic = self._cros_grad.deriv2(self._models)
        
        
        # Numerical solution
        d2c_dm1 = np.zeros([len(self._m1),len(self._m1)]) #H11
        d2c_dm2 = d2c_dm1.copy() #H22
        d_dm1_dc_dm2 = d2c_dm1.copy() #H12
        d_dm2_dc_dm1 = d2c_dm1.copy() #H21        
                
        
        for i in range(len(self._m1)):
            for j in range(len(self._m1)):
                
                # H11 (d2c_dm1)
                m1_tmp = np.copy(self._m1)
                if m1_tmp[j] == 0:
                    m1_tmp[j] = self._delta_m
                    J1 = self._cros_grad.deriv(np.r_[m1_tmp, self._m2])
                    d2c_dm1[i,j] = (J1[i] - J0[i]) / self._delta_m
                else:
                    m1_tmp[j] = m1_tmp[j] * self._delta_per
                    J1 = self._cros_grad.deriv(np.r_[m1_tmp, self._m2])
                    d2c_dm1[i,j] = (J1[i] - J0[i]) / self._delta_m
        
        
                # H22 (d2c_dm2)
                m2_tmp = np.copy(self._m2)
                if m2_tmp[j] == 0:
                    m2_tmp[j] = self._delta_m
                    J1 = self._cros_grad.deriv(np.r_[self._m1, m2_tmp])
                    d2c_dm2[i,j] = (J1[i+n] - J0[i+n]) / self._delta_m
                else:
                    m2_tmp[j] = m2_tmp[j] * self._delta_per
                    J1 = self._cros_grad.deriv(np.r_[self._m1, m2_tmp])
                    d2c_dm2[i,j] = (J1[i+n] - J0[i+n]) / self._delta_m
        
        
                # H12 (d_dm2_dc_dm1)
                m2_tmp = np.copy(self._m2)
                if m2_tmp[j] == 0:
                    m2_tmp[j] = self._delta_m
                    J1 = self._cros_grad.deriv(np.r_[self._m1, m2_tmp])
                    d_dm2_dc_dm1[i,j] = (J1[i] - J0[i]) / self._delta_m
                else:
                    m2_tmp[j] = m2_tmp[j] * self._delta_per
                    J1 = self._cros_grad.deriv(np.r_[self._m1, m2_tmp])
                    d_dm2_dc_dm1[i,j] = (J1[i] - J0[i]) / self._delta_m
        
        
                # H21 (d_dm1_dc_dm2)
                m1_tmp = np.copy(self._m1)
                if m1_tmp[j] == 0:
                    m1_tmp[j] = self._delta_m
                    J1 = self._cros_grad.deriv(np.r_[m1_tmp, self._m2])
                    d_dm1_dc_dm2[i,j] = (J1[i+n] - J0[i+n]) / self._delta_m
                else:
                    m1_tmp[j] = m1_tmp[j] * self._delta_per
                    J1 = self._cros_grad.deriv(np.r_[m1_tmp, self._m2])
                    d_dm1_dc_dm2[i,j] = (J1[i+n] - J0[i+n]) / self._delta_m        
            
        
        
        temp1 = np.vstack((d2c_dm1,d_dm1_dc_dm2))
        temp2 = np.vstack((d_dm2_dc_dm1, d2c_dm2))
        H_numerical = np.hstack((temp1,temp2))
        
        # convert sparse array to dense array
        H_analytic = sp.csr_matrix.toarray(H_analytic)
        
        
        self.assertAlmostEqual(
            np.all(H_analytic), 
            np.all(H_numerical), 
            places=decimal_digit
            )
        
        

if __name__ == "__main__":
    unittest.main()        
        
        
        
        
        