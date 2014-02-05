#! /bin/bash

locale-gen en_US en_US.UTF-8 hu_HU hu_HU.UTF-8 > output.t
dpkg-reconfigure locales >> output.t

sudo apt-get update  >> output.t
echo " "
echo " "
echo "   ============================================"
echo "   | Installing packages form package manager |"
echo "   ============================================"
echo " "
echo " "

sudo apt-get -y install aptitude >> output.t

packages=(gcc gfortran git libopenmpi-dev python-pip python-dev git flex bison cmake vim cython ipython python-scipy python-numpy python-nose python-pip python-matplotlib python-vtk python-h5py libmumps-ptscotch-4.10.0 libmumps-ptscotch-dev libblas-dev liblapack-dev )


for item in ${packages[*]}
do
    printf "     %-30s\n" $item

done

for item in ${packages[*]}
do
    tput cuu1
done

for item in ${packages[*]}
do
    sudo aptitude -y install $item >> output.t
    printf "     %-30s              %-4s\n" $item done
done


echo " "
echo " "
echo "   ====================================="
echo "   | Installing extra Python libraries |"
echo "   ====================================="
echo " "
echo " "


pipPackages=(mpi4py pymumps)

for item in ${pipPackages[*]}
do
    printf "     %-30s\n" $item
done

for item in ${pipPackages[*]}
do
    tput cuu1
done

for item in ${pipPackages[*]}
do
        sudo pip install $item >> output.t
        printf "     %-30s              %-4s\n" $item done
done

Upgrade=(scipy numpy ipython)

for item in ${Upgrade[*]}
do
    printf "     %-8s%-7s\n" $item upgrade

done

for item in ${Upgrade[*]}
do
    tput cuu1
done

for item in ${Upgrade[*]}
do
    sudo pip install $item --upgrade >> output.t
    printf "     %-8s%-7s                             %-4s\n" $item upgrade done
done



echo " "
echo " "
echo "   ====================="
echo "   | Installing SimPEG |"
echo "   ====================="
echo " "
echo " "
cd ~


git clone https://github.com/simpeg/simpeg.git >> output.t
cd simpeg/SimPEG/
python setup.py >> output.t
cd ~

mkdir petsc
cd petsc

echo " "
echo " "
echo "   ===================="
echo "   | Installing PETSc |"
echo "   ===================="
echo " "
echo " "
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.4.3.tar.gz

tar -zxf petsc-3.4.3.tar.gz

cd petsc-3.4.3

./configure --with-debugging=no  --dowload-mpich=yes --download-blacs=yes --download-f-blas-lapack=yes --download-scalapack=yes --download-mumps=yes --download-ml=yes --download-spooles=yes --download-hypre=yes --dowload-trilinos=yes --download-metis=yes --download-parmetis=yes --download-umfpack=yes --download-ptscotch=yes  --download-superlu=yes --download-superlu_dist=yes --download-essl=yes --download-eucild=yes  --download-spai=yes  --download-mpi4py=yes --download-petsc4py=yes --download-scientificpython=yes


echo "export PETSC_DIR=/home/${USER}/petsc/petsc-3.4.3" >> ~/.bashrc
echo "export PETSC_ARCH=arch-linux2-c-opt" >> ~/.bashrc
export PETSC_DIR=/home/${USER}/petsc/petsc-3.4.3
export PETSC_ARCH=arch-linux2-c-opt
. ~/.bashrc

make PETSC_DIR=/home/${USER}/petsc/petsc-3.4.3 PETSC_ARCH=arch-linux2-c-opt all
make PETSC_DIR=/home/${USER}/petsc/petsc-3.4.3 PETSC_ARCH=arch-linux2-c-opt test

cd ~/petsc
echo " "
echo " "
echo "   ======================="
echo "   | Installing PETSc4PY |"
echo "   ======================="
echo " "
echo " "
git clone https://bitbucket.org/petsc/petsc4py.git
cd petsc4py/
python setup.py build >> output.t
python setup.py install --prefix=~/petsc >> output.t

echo "export PYTHONPATH=~/petsc/lib/python2.7/site-packages:/home/$USER/simpeg:${PYTHONPATH}" >> ~/.bashrc

cd ~
source ~/.bashrc
