import os
try:
    import SimPEG
except ImportError, e:
    os.system('git clone https://github.com/simpeg/simpeg.git')
    os.system('mv simpeg/SimPEG temp')
    os.system('rm -rf simpeg')
    os.system('mv temp SimPEG')
    # os.system('mkdir simp')
    print os.path.abspath('.')
    # os.system('cp -r SimPEG simpegflow/docs/examples/SimPEG')
