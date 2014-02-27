import os
try:
    import SimPEG
except ImportError, e:
    os.system('git clone https://github.com/simpeg/simpeg.git')
    os.system('python simpeg/setup.py install')

    os.system('mv simpeg/SimPEG temp')
    os.system('rm -rf simpeg')
    os.system('mv temp SimPEG')
