# ------------ Dummy codes for older version ------------ #
# Deprecated with the latest versions in geoana
# ------------------------------------------------------- #

# try:
#     from simpegEM1D.m_rTE_Fortran import rte_fortran
# except ImportError as e:
#     rte_fortran = None

# def rTEfunfwd(n_layer, f, lamda, sig, chi, thick, halfspace_switch):
#     """
#         Compute reflection coefficients for Transverse Electric (TE) mode.
#         Only one for loop for multiple layers.

#         Parameters
#         ----------
#         n_layer : int
#             The number layers
#         f : complex, ndarray
#             Frequency (Hz); size = (n_frequency x n_filter)
#         lamda : complex, ndarray
#             Frequency (Hz); size = (n_frequency x n_filter)
#         sig: compelx, ndarray
#             Conductivity (S/m); size = (n_layer x n_frequency x n_filter)
#         chi: compelx, ndarray
#             Susceptibility (SI); size = (n_layer,)
#         depth: float, ndarray
#             Top boundary of the layers; size = (n_ayer,)
#         halfspace_switch: bool
#             Switch for halfspace

#         Returns
#         -------
#         rTE: compex, ndarray
#             Reflection coefficients;
#             size = (n_frequency x n_lamba)
#     """

#     n_frequency, n_filter = lamda.shape

#     Mtemp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     M1sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     w = 2*np.pi*f

#     rTE = np.zeros((n_frequency, n_filter), dtype=complex)
#     utemp0 = np.zeros((n_frequency, n_filter), dtype=complex)
#     utemp1 = np.zeros((n_frequency, n_filter), dtype=complex)
#     const = np.zeros((n_frequency, n_filter), dtype=complex)

#     utemp0 = lamda
#     utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0, :, :])*sig[0, :, :])
#     const = mu_0*utemp1/(mu_0*(1+chi[0, :, :])*utemp0)

#     Mtemp00 = 0.5*(1+const)
#     Mtemp10 = 0.5*(1-const)
#     Mtemp01 = 0.5*(1-const)
#     Mtemp11 = 0.5*(1+const)

#     # may be store these and reuse for sensitivity?
#     M00 = []
#     M10 = []
#     M01 = []
#     M11 = []

#     M0sum00 = Mtemp00
#     M0sum10 = Mtemp10
#     M0sum01 = Mtemp01
#     M0sum11 = Mtemp11

#     if halfspace_switch:

#         M1sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#         M1sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#         M1sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#         M1sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#         M1sum00 = M0sum00
#         M1sum10 = M0sum10
#         M1sum01 = M0sum01
#         M1sum11 = M0sum11

#     else:

#         for j in range(n_layer-1):
#             utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j, :, :])*sig[j, :, :])
#             utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1, :, :])*sig[j+1, :, :])
#             const = mu_0*(1+chi[j, :, :])*utemp1/(mu_0*(1+chi[j+1, :, :])*utemp0)

#             h0 = thick[j]

#             Mtemp00 = 0.5*(1.+const)*np.exp(-2.*utemp0*h0)
#             Mtemp10 = 0.5*(1.-const)
#             Mtemp01 = 0.5*(1.-const)*np.exp(-2.*utemp0*h0)
#             Mtemp11 = 0.5*(1.+const)

#             M1sum00 = M0sum00*Mtemp00 + M0sum01*Mtemp10
#             M1sum10 = M0sum10*Mtemp00 + M0sum11*Mtemp10
#             M1sum01 = M0sum00*Mtemp01 + M0sum01*Mtemp11
#             M1sum11 = M0sum10*Mtemp01 + M0sum11*Mtemp11

#             M0sum00 = M1sum00
#             M0sum10 = M1sum10
#             M0sum01 = M1sum01
#             M0sum11 = M1sum11

#     rTE = M1sum01/M1sum11

#     return rTE


# def matmul(a00, a10, a01, a11, b00, b10, b01, b11):
#     """
#         Compute 2x2 matrix mutiplication in vector way
#         C = A*B
#         C = [a00   a01] * [b00   b01] = [c00   c01]
#         [a10   a11]   [b10   b11]   [c10   c11]
#     """

#     c00 = a00*b00 + a01*b10
#     c10 = a10*b00 + a11*b10
#     c01 = a00*b01 + a01*b11
#     c11 = a10*b01 + a11*b11

#     return c00, c10, c01, c11



# # TODO: make this to take a vector rather than a single frequency
# def rTEfunjac(n_layer, f, lamda, sig, chi, thick, halfspace_switch):
#     """
#         Compute sensitivity of reflection coefficients for
#         Transverse Electric (TE) mode with regard to conductivity

#         Parameters
#         ----------
#         n_layer : int
#             The number layers
#         f : complex, ndarray
#             Frequency (Hz); size = (n_frequency x n_finlter)
#         lamda : complex, ndarray
#             Frequency (Hz); size = (n_frequency x n_finlter)
#         sig: complex, ndarray
#             Conductivity (S/m); size = (n_layer x 1)
#         chi: compelx, ndarray
#             Susceptibility (SI); size = (n_layer x 1)
#         depth: float, ndarray
#             Top boundary of the layers
#         halfspace_switch: bool
#             Switch for halfspace

#         Returns
#         -------
#         rTE: compex, ndarray
#             Derivative of reflection coefficients;
#             size = (n_frequency x n_layer x n_finlter)
#     """
#     # Initializing arrays
#     n_frequency, n_filter = lamda.shape

#     Mtemp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     Mtemp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     M1sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M1sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     M0sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M0sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M0sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     M0sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     dMtemp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dMtemp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dMtemp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dMtemp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     dj0temp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj0temp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj0temp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj0temp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     dj1temp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj1temp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj1temp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#     dj1temp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#     w = 2*np.pi*f

#     rTE = np.zeros((n_frequency, n_filter), dtype=complex)
#     drTE = np.zeros((n_layer, n_frequency, n_filter), dtype=complex)
#     utemp0 = np.zeros((n_frequency, n_filter), dtype=complex)
#     utemp1 = np.zeros((n_frequency, n_filter), dtype=complex)
#     const = np.zeros((n_frequency, n_filter), dtype=complex)

#     utemp0 = lamda
#     utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0, :, :])
#     const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

#     # Compute M1
#     Mtemp00 = 0.5*(1+const)
#     Mtemp10 = 0.5*(1-const)
#     Mtemp01 = 0.5*(1-const)
#     Mtemp11 = 0.5*(1+const)

#     utemp0 = lamda
#     utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0, :, :])
#     const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

#     # Compute dM1du1
#     dj0Mtemp00 =  0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
#     dj0Mtemp10 = -0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
#     dj0Mtemp01 = -0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
#     dj0Mtemp11 =  0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))

#     # TODO: for computing Jacobian
#     M00 = []
#     M10 = []
#     M01 = []
#     M11 = []

#     dJ00 = []
#     dJ10 = []
#     dJ01 = []
#     dJ11 = []

#     M00.append(Mtemp00)
#     M01.append(Mtemp01)
#     M10.append(Mtemp10)
#     M11.append(Mtemp11)

#     M0sum00 = Mtemp00.copy()
#     M0sum10 = Mtemp10.copy()
#     M0sum01 = Mtemp01.copy()
#     M0sum11 = Mtemp11.copy()

#     if halfspace_switch or n_layer == 1:

#         M1sum00 = M0sum00.copy()
#         M1sum10 = M0sum10.copy()
#         M1sum01 = M0sum01.copy()
#         M1sum11 = M0sum11.copy()

#     else:

#         for j in range(n_layer-1):

#             dJ_10Mtemp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ_10Mtemp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ_10Mtemp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ_10Mtemp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#             dJ01Mtemp00 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ01Mtemp10 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ01Mtemp01 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ01Mtemp11 = np.zeros((n_frequency, n_filter), dtype=complex)

#             utemp0  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j])*sig[j, :, :])
#             utemp1  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1])*sig[j+1, :, :])
#             const = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0)

#             h0 = thick[j]

#             Mtemp00 = 0.5*(1.+ const)*np.exp(-2.*utemp0*h0)
#             Mtemp10 = 0.5*(1.- const)
#             Mtemp01 = 0.5*(1.- const)*np.exp(-2.*utemp0*h0)
#             Mtemp11 = 0.5*(1.+ const)

#             M1sum00, M1sum10, M1sum01, M1sum11 = matmul(
#                 M0sum00, M0sum10, M0sum01, M0sum11,
#                 Mtemp00, Mtemp10, Mtemp01, Mtemp11
#             )

#             M0sum00 = M1sum00
#             M0sum10 = M1sum10
#             M0sum01 = M1sum01
#             M0sum11 = M1sum11

#             # TODO: for Computing Jacobian

#             dudsig = 0.5*1j*w*mu_0*(1+chi[j])/utemp0

#             if j == 0:

#                 const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
#                 const1b = const1a*utemp0

#                 dj1Mtemp00 = -0.5*const1a*np.exp(-2.*utemp0*h0)-h0*(1+const1b)*np.exp(-2.*utemp0*h0)
#                 dj1Mtemp10 =  0.5*const1a
#                 dj1Mtemp01 =  0.5*const1a*np.exp(-2.*utemp0*h0)-h0*(1-const1b)*np.exp(-2.*utemp0*h0)
#                 dj1Mtemp11 = -0.5*const1a

#                 # Compute dM1dm1*M2
#                 dJ_10Mtemp00, dJ_10Mtemp10, dJ_10Mtemp01, dJ_10Mtemp11 = matmul(
#                     dj0Mtemp00, dj0Mtemp10, dj0Mtemp01, dj0Mtemp11,
#                     Mtemp00, Mtemp10, Mtemp01, Mtemp11
#                 )

#                 # Compute M1*dM2dm1
#                 dJ01Mtemp00, dJ01Mtemp10, dJ01Mtemp01, dJ01Mtemp11 = matmul(
#                     M00[j], M10[j], M01[j], M11[j], dj1Mtemp00,
#                     dj1Mtemp10, dj1Mtemp01, dj1Mtemp11
#                 )

#                 dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
#                 dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
#                 dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
#                 dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

#             else:

#                 h_1 = thick[j-1]
#                 utemp_1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j-1])*sig[j-1])
#                 const0 = mu_0*(1+chi[j-1])/(mu_0*(1+chi[j])*utemp_1)

#                 dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
#                 dj0Mtemp10 = -0.5*(const0)
#                 dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
#                 dj0Mtemp11 =  0.5*(const0)

#                 const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
#                 const1b = const1a*utemp0

#                 dj1Mtemp00 = -0.5*const1a*np.exp(-2.*utemp0*h0)-h0*(1+const1b)*np.exp(-2.*utemp0*h0)
#                 dj1Mtemp10 =  0.5*const1a
#                 dj1Mtemp01 =  0.5*const1a*np.exp(-2.*utemp0*h0)-h0*(1-const1b)*np.exp(-2.*utemp0*h0)
#                 dj1Mtemp11 = -0.5*const1a

#                 # Compute dMjdmj*Mj+1
#                 dJ_10Mtemp00, dJ_10Mtemp10, dJ_10Mtemp01, dJ_10Mtemp11 = matmul(
#                     dj0Mtemp00, dj0Mtemp10, dj0Mtemp01, dj0Mtemp11,
#                     Mtemp00, Mtemp10, Mtemp01, Mtemp11
#                 )

#                 # Compute Mj*dMj+1dmj
#                 dJ01Mtemp00, dJ01Mtemp10, dJ01Mtemp01, dJ01Mtemp11 = matmul(
#                     M00[j], M10[j], M01[j], M11[j],
#                     dj1Mtemp00, dj1Mtemp10, dj1Mtemp01, dj1Mtemp11
#                 )

#                 dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
#                 dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
#                 dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
#                 dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

#             M00.append(Mtemp00)
#             M01.append(Mtemp01)
#             M10.append(Mtemp10)
#             M11.append(Mtemp11)

#     # rTE = M1sum01/M1sum11

#     if halfspace_switch or n_layer == 1:

#         utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
#         dudsig = 0.5*1j*w*mu_0*(1+chi[0])/utemp0

#         dJ1sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#         dJ1sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#         dJ1sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#         dJ1sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#         dJ1sum00 = dudsig*dj0Mtemp00
#         dJ1sum10 = dudsig*dj0Mtemp10
#         dJ1sum01 = dudsig*dj0Mtemp01
#         dJ1sum11 = dudsig*dj0Mtemp11

#         drTE = dJ1sum01/M1sum11 - M1sum01/(M1sum11**2)*dJ1sum11

#     else:

#         # j = n_layer
#         utemp0 = (
#             np.sqrt(lamda**2+1j*w*mu_0*(1+chi[n_layer-1])*sig[n_layer-1, :, :])
#         )
#         dudsig = 0.5*1j*w*mu_0*(1+chi[n_layer-1])/utemp0

#         h_1 = thick[n_layer-2]

#         utemp_1 = (
#             np.sqrt(lamda**2+1j*w*mu_0*(1+chi[n_layer-2])*sig[n_layer-2, :, :])
#         )
#         const0 = mu_0*(1+chi[n_layer-2])/(mu_0*(1+chi[n_layer-1])*utemp_1)

#         dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
#         dj0Mtemp10 = -0.5*(const0)
#         dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
#         dj0Mtemp11 =  0.5*(const0)

#         dJ_10Mtemp00 = dj0Mtemp00
#         dJ_10Mtemp10 = dj0Mtemp10
#         dJ_10Mtemp01 = dj0Mtemp01
#         dJ_10Mtemp11 = dj0Mtemp11

#         dJ00.append(dudsig*dJ_10Mtemp00)
#         dJ10.append(dudsig*dJ_10Mtemp10)
#         dJ01.append(dudsig*dJ_10Mtemp01)
#         dJ11.append(dudsig*dJ_10Mtemp11)

#         for i in range(n_layer):

#             dJ0sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ0sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ0sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ0sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#             dJ1sum00 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ1sum10 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ1sum01 = np.zeros((n_frequency, n_filter), dtype=complex)
#             dJ1sum11 = np.zeros((n_frequency, n_filter), dtype=complex)

#             if i == 0:

#                 for j in range(n_layer-2):

#                     if j == 0:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ00[i], dJ10[i], dJ01[i], dJ11[i],
#                             M00[j+2], M10[j+2], M01[j+2], M11[j+2]
#                         )

#                     else:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             M00[j+2], M10[j+2], M01[j+2], M11[j+2]
#                         )

#                     dJ0sum00 = dJ1sum00
#                     dJ0sum10 = dJ1sum10
#                     dJ0sum01 = dJ1sum01
#                     dJ0sum11 = dJ1sum11

#             elif (i > 0) & (i < n_layer-1):

#                 dJ0sum00 = M00[0]
#                 dJ0sum10 = M10[0]
#                 dJ0sum01 = M01[0]
#                 dJ0sum11 = M11[0]

#                 for j in range (n_layer-2):

#                     if j==i-1:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             dJ00[i], dJ10[i], dJ01[i], dJ11[i]
#                         )

#                     elif j < i-1:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             M00[j+1], M10[j+1], M01[j+1], M11[j+1]
#                         )

#                     elif j > i-1:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             M00[j+2], M10[j+2], M01[j+2], M11[j+2]
#                         )

#                     dJ0sum00 = dJ1sum00
#                     dJ0sum10 = dJ1sum10
#                     dJ0sum01 = dJ1sum01
#                     dJ0sum11 = dJ1sum11

#             elif i == n_layer-1:

#                 dJ0sum00 = M00[0]
#                 dJ0sum10 = M10[0]
#                 dJ0sum01 = M01[0]
#                 dJ0sum11 = M11[0]

#                 for j in range(n_layer-1):

#                     if j < n_layer-2:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             M00[j+1], M10[j+1], M01[j+1], M11[j+1]
#                         )

#                     elif j == n_layer-2:

#                         dJ1sum00, dJ1sum10, dJ1sum01, dJ1sum11 = matmul(
#                             dJ0sum00, dJ0sum10, dJ0sum01, dJ0sum11,
#                             dJ00[i], dJ10[i], dJ01[i], dJ11[i]
#                         )

#                     dJ0sum00 = dJ1sum00
#                     dJ0sum10 = dJ1sum10
#                     dJ0sum01 = dJ1sum01
#                     dJ0sum11 = dJ1sum11

#             drTE[i, :] = dJ1sum01/M1sum11 - M1sum01/(M1sum11**2)*dJ1sum11

#     return drTE
    # Still worthwhile to output both?
    # return rTE, drTE


    # ------------ Dummy codes for older version ------------ #
    # This can be used in later use for handling on-time data.
    # ------------------------------------------------------- #

    # def setWaveform(self, **kwargs):
    #     """
    #         Set parameters for Src Waveform
    #     """
    #     # TODO: this is hp is only valid for Circular loop system
    #     self.hp = self.I/self.a*0.5

    #     self.toff = kwargs['toff']
    #     self.waveform = kwargs['waveform']
    #     self.waveformDeriv = kwargs['waveformDeriv']
    #     self.tconv = kwargs['tconv']

    # def projectFields(self, u):
    #     """
    #         Transform frequency domain responses to time domain responses
    #     """
    #     # Case1: Compute frequency domain reponses right at filter coefficient values
    #     if self.switchInterp == False:
    #         # Src waveform: Step-off
    #         if self.wave_type == 'stepoff':
    #             if self.rx_type == 'Bz':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp, f0 = transFilt(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i], f0 = transFilt(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

    #             elif self.rx_type == 'dBzdt':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp = -transFiltImpulse(u, self.wt,self.tbase, self.frequency*2*np.pi, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i] = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

    #         # Src waveform: General (it can be any waveform)
    #         # We evaluate this with time convolution
    #         elif self.wave_type == 'general':
    #             # Compute EM responses
    #             if u.size == self.n_frequency:
    #                 # TODO: write small code which compute f at t = 0
    #                 f, f0 = transFilt(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
    #                 fDeriv = -transFiltImpulse(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

    #                 if self.rx_type == 'Bz':

    #                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')

    #                     # TODO: make it as an opition #2
    #                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                     # respint = interp1d(self.tconv, resp2, 'linear')

    #                     resp = respint(self.time)

    #                 if self.rx_type == 'dBzdt':
    #                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')
    #                     resp = respint(self.time)

    #             # Compute EM sensitivities
    #             else:

    #                 resp = np.zeros((self.n_time, self.n_layer))
    #                 for i in range (self.n_layer):

    #                     f, f0 = transFilt(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
    #                     fDeriv = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

    #                     if self.rx_type == 'Bz':

    #                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')

    #                         # TODO: make it as an opition #2
    #                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                         # respint = interp1d(self.tconv, resp2, 'linear')

    #                         resp[:,i] = respint(self.time)

    #                     if self.rx_type == 'dBzdt':
    #                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')
    #                         resp[:,i] = respint(self.time)

    #     # Case2: Compute frequency domain reponses in logarithmic then intepolate
    #     if self.switchInterp == True:
    #         # Src waveform: Step-off
    #         if self.wave_type == 'stepoff':
    #             if self.rx_type == 'Bz':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp, f0 = transFiltInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i], f0 = transFiltInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

    #             elif self.rx_type == 'dBzdt':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i] = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

    #         # Src waveform: General (it can be any waveform)
    #         # We evaluate this with time convolution
    #         elif self.wave_type == 'general':
    #             # Compute EM responses
    #             if u.size == self.n_frequency:
    #                 # TODO: write small code which compute f at t = 0
    #                 f, f0 = transFiltInterp(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
    #                 fDeriv = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

    #                 if self.rx_type == 'Bz':

    #                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')

    #                     # TODO: make it as an opition #2
    #                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                     # respint = interp1d(self.tconv, resp2, 'linear')

    #                     resp = respint(self.time)

    #                 if self.rx_type == 'dBzdt':
    #                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')
    #                     resp = respint(self.time)

    #             # Compute EM sensitivities
    #             else:

    #                 resp = np.zeros((self.n_time, self.n_layer))
    #                 for i in range (self.n_layer):

    #                     f, f0 = transFiltInterp(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
    #                     fDeriv = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

    #                     if self.rx_type == 'Bz':

    #                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')

    #                         # TODO: make it as an opition #2
    #                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                         # respint = interp1d(self.tconv, resp2, 'linear')

    #                         resp[:,i] = respint(self.time)

    #                     if self.rx_type == 'dBzdt':
    #                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')
    #                         resp[:,i] = respint(self.time)

    #     return mu_0*resp
