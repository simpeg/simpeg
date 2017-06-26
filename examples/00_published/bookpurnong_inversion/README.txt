Bookpurnong Inversion
=====================

These are the data files required to reproduce the Bookpurnong Inversion results published in

Heagy, L.J., R. Cockett, S. Kang, G.K. Rosenkjaer, D.W. Oldenburg,
2017, A framework for simulation and inversion in electromagnetics.
Computers & Geosciences


The original data are in the directory `bookpurnong_inversion` and are available from:

https://storage.googleapis.com/simpeg/bookpurnong/bookpurnong.tar.gz

The code used to run the inversions in the paper is available at http://docs.simpeg.xyz.

Contents
--------

Input files:
    - MurrayRiver.txt: Murray river path
    - skytem_hm.wf: SkyTEM waveform
    - skytem_hm.tc: SkyTEM time channels (center)

Output files:
    - booky_resolve.hdf5: downsampled RESOLVE data (output of loadbooky.py)
    - booky_skytem.hdf5: downsampled SkyTEM data (output of loadbooky.py)
    - dobs_re_final.npy: Observed data from 1D stitched inversion (RESOLVE)
    - dpred_re_final.npy: Predicted data from the 1D stitched inversion (RESOLVE)
    - mopt_re_final.npy: Recovered model at each sounding (log sigma) from the 1D stitched inversion of RESOLVE data


bookpurnong_data
^^^^^^^^^^^^^^^^

The RESOLVE and SkyTEM data collected over Bookpurnong have been made available with permission from CSIRO. Please acknowledge CSIRO if using these data in a presentation, publication, etc.

Two data sets are included in this distribution, RESOLVE data collected in 2008, and SkyTEM (High Moment) data collected in 2006.

For an example of how to load and plot the data, please see: http://docs.simpeg.xyz

- 8044_Boopurnong.HDR : RESOLVE header file for the 2008 Bookpurnong survey
- Bookpurnong_Resolve_Exported.XYZ : RESOLVE data collected in 2008
- Bookpurnong_SkyTEM.HDR : SkyTEM header file for the 2006 Bookpurnong survey
- SK655CS_Bookpurnong_ZX_HM_TxInc_newDTM.txt : SkyTEM high moment data collected in 2006




