#! /bin/bash
sudo aptitude -y update
sudo aptitude -y upgrade
sudo aptitude -y install gcc gfortran git libopenmpi-dev python-pip python-dev
sudo aptitude -y install ipython python-scipy python-numpy python-nose python-pip python-matplotlib
sudo aptitude -y install libmumps-ptscotch-4.10.0 libmumps-ptscotch-dev
sudo aptitude -y install libblas-dev liblapack-dev

sudo pip install mpi4py
sudo pip install pymumps

sudo pip install scipy --upgrade
sudo pip install numpy --upgrade
sudo pip install ipython --upgrade

git clone https://bitbucket.org/rcockett/simpeg.git
cd simpeg/SimPEG/
python setup.py
cd ~

echo export PYTHONPATH=/home/$USER/simpeg/ >> .bashrc
source .profile