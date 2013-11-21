#! /bin/bash
sudo aptitude -y update
sudo aptitude -y upgrade
sudo aptitude -y install gcc gfortran git libopenmpi-dev python-pip python-dev
sudo aptitude -y install ipython python-scipy python-numpy 
sudo aptitude -y install python-matplotlib python-nose python-pip
sudo aptitude -y install libmumps-ptscotch-4.10.0 libmumps-ptscotch-dev
sudo pip install mpi4py
sudo pip install pymumps
git clone https://dwfmarchant@bitbucket.org/rcockett/simpeg.git
cd simpeg/SimPEG/
python setup.py
cd ~