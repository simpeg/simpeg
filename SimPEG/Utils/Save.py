import numpy as np
import time
import re

try:
    import h5py
except Exception, e:
    print 'Warning: SimPEG.Utils.Save needs h5py to be installed.'


SAVEABLES = {}

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    atoi = lambda text: int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def preIteration(group):
    group.attrs['complete'] = False
    group.attrs['time'] = time.time()

def postIteration(group):
    group.attrs['time'] = time.time() - group.attrs['time']
    group.attrs['date'] = time.ctime()
    group.attrs['complete'] = True

class SimPEGTable:
    """
        This is a wrapper class on the HDF5 file.
    """
    def __init__(self, name, mode='a'):
        if '.hdf5' not in name:
            name += '.hdf5'
        self.f = h5py.File(name,  mode)
        self.root = hdf5Group(self,self.f)

        self.inversions = hdf5InversionGroup(self,self.root.addGroup('inversions',soft=True))

    def show(self): self.root.show()

    def saveInversion(self, invObj):

        # Create a new inversion anytime this is run.
        def _startup_hdf5_inv(invObj, m0):
            node = self.inversions.addGroup('%d'%self.inversions.numChildren)
            saveSavable(invObj,node.addGroup('rebuild'))
            results = node.addGroup('results')
            preIteration(results)
            invObj._invNode = results
            self.f.flush()
        invObj.hook(_startup_hdf5_inv, overwrite=True)

        # At the start of every iteration we will create a inversion iteration node.
        def _doStartIteration_hdf5_inv(invObj):
            invObj._invNodeIt = invObj._invNode.addGroup('%d'%(invObj._iter+1))
            preIteration(invObj._invNodeIt)
        invObj.hook(_doStartIteration_hdf5_inv, overwrite=True)

        def _doEndIteration_hdf5_inv(invObj):
            invObj.save(invObj._invNodeIt)
            postIteration(invObj._invNodeIt)
            self.f.flush()
        invObj.hook(_doEndIteration_hdf5_inv, overwrite=True)

        # Delete all iterates that did not finish.
        def _finish_hdf5_inv(invObj):
            postIteration(invObj._invNode)
            for it in invObj._invNode:
                if not it.attrs['complete']:
                    del self.f[it.path]
            del invObj._invNode
            self.f.flush()
        invObj.hook(_finish_hdf5_inv, overwrite=True)

        def _doStartIteration_hdf5_opt(optObj):
            optObj._optNodeIt = optObj.parent._invNode.addGroup('%d.%d'%(optObj.parent._iter, optObj._iter))
            preIteration(optObj._optNodeIt)
        invObj.opt.hook(_doStartIteration_hdf5_opt, overwrite=True)

        def _doEndIteration_hdf5_opt(optObj, xt):
            optObj.save(optObj._optNodeIt)
            postIteration(optObj._optNodeIt)
            self.f.flush()
        invObj.opt.hook(_doEndIteration_hdf5_opt, overwrite=True)



class hdf5Group(object):
    """Has some low level support for wrapping the native HDF5-Group class"""

    def __init__(self, T, groupNode):
        self.T = T
        # check if you are inputing a hdf5Group rather than a raw node, and act accordingly
        if issubclass(groupNode.__class__, hdf5Group):
            self.node = groupNode.node
        else:
            self.node = groupNode

        self.childClass = hdf5Group
        self.parentClass = hdf5Group

    @property
    def children(self):
        """Children names in a list

            Use obj[name] to return the actual node.
        """
        myChildren = [c for c in self.node]
        myChildren.sort(key=natural_keys)
        return myChildren

    @property
    def numChildren(self):
        """Returns the len(children)"""
        return len(self.children)

    @property
    def parent(self):
        """Returns parent node"""
        return self.parentClass(self.T, self.node.parent)

    @property
    def name(self):
        return self.path.split('/')[-1]

    @property
    def path(self):
        """Returns the root path of the group"""
        return self.node.name

    @property
    def attrs(self):
        """Returns a list of attributes in the group"""
        return self.node.attrs

    def addGroup(self, name, soft=False):
        """Adds a child group to the current node."""
        if name in self.children and soft:
            return self[name]
        assert name not in self.children, 'Already a child called: '+self.path+'/'+name
        return self.childClass(self.T, self.node.create_group(name))

    def setArray(self, name, data):
        a = self.node.create_dataset(name, data.shape)
        a[...] = data
        return a

    def __getitem__(self, val):
        if type(val) is int:
            val = self.children[val]
        child = self.node[val]
        if type(child) is h5py.Group:
            child = self.childClass(self.T, child)
        return child

    def __contains__(self, key):
        return key in self.children

    def show(self, pad='', maxDepth=1, depth=0):
        """
            Recursively show the structure of the database.

            For example::

                <hdf5InversionGroup group "/inversions" (1 member)>
                    - <hdf5Inversion group "/inversions/0" (4 members)>
                        - <hdf5InversionIteration group "/inversions/0/0.0" (3 members)>
                        - <hdf5InversionIteration group "/inversions/0/0.1" (3 members)>
                        - <hdf5InversionIteration group "/inversions/0/0.2" (3 members)>
                        - <hdf5InversionIteration group "/inversions/0/0.3" (3 members)>
        """
        s = self.__str__()
        pad += ' '*4
        if maxDepth <= 0: print s
        if depth >= maxDepth: return s

        for c in self.children:
            if issubclass(self[c].__class__, hdf5Group):
                s += '\n%s- %s' % (pad, self[c].show(pad=pad,depth=depth+1,maxDepth=maxDepth))
            else:
                s += '\n%s- %s' % (pad, self[c].__str__())
        if depth is 0:
            print s
        else:
            return s

    def __str__(self):
        return '<%s "%s" (%d member%s, attrs=[%s])>' % (self.__class__.__name__, self.path, self.numChildren, '' if self.numChildren == 1 else 's',', '.join([a for a in self.attrs]))



class hdf5InversionGroup(hdf5Group):
    def __init__(self, T, groupNode):
        hdf5Group.__init__(self, T, groupNode)
        self.childClass = hdf5Inversion

class hdf5Inversion(hdf5Group):
    def __init__(self, T, groupNode):
        hdf5Group.__init__(self, T, groupNode)
        self.parentClass = hdf5InversionGroup
        self.childClass = hdf5InversionResults

    def rebuild(self):
        return loadSavable(self['rebuild'])

    @property
    def results(self): return self['results']


class hdf5InversionResults(hdf5Group):
    def __init__(self, T, groupNode):
        hdf5Group.__init__(self, T, groupNode)
        self.parentClass = hdf5Inversion
        self.childClass = hdf5InversionIteration

class hdf5InversionIteration(hdf5Group):
    def __init__(self, T, groupNode):
        hdf5Group.__init__(self, T, groupNode)
        self.parentClass = hdf5InversionResults



class Savable(type):
    def __new__(cls, name, bases, attrs):
        __init__ = attrs['__init__']
        def init_wrapper(self, *args, **kwargs):
            self._args_init = args
            self._kwargs_init = kwargs
            return __init__(self, *args,**kwargs)
        attrs['__init__'] = init_wrapper

        newClass = super(Savable, cls).__new__(cls, name, bases, attrs)
        SAVEABLES[name] = newClass
        return newClass


def saveSavable(obj, group, debug=False):
    """
        This creates softlinks if _savable exists in children object.

        The first object is always created.
    """
    assert type(obj.__class__) is Savable, 'Can only save objects that are Savable objects.'

    def doSave(grp, name, val):
        if debug: print name, val
        if type(val.__class__) is Savable:
            link = getattr(val,'_savable',None)
            if link is not None:
                group.node[name] = h5py.SoftLink(link.path)
                if debug: 'Created a softlink path to %s' % link.path
            else:
                subgrp = grp.addGroup(name)
                saveSavable(val, subgrp, debug=debug)
        elif type(val) in [list, tuple]:
            # Split up, and save each element
            for i, v in enumerate(val):
                doSave(grp, name + '[%d]'%i, v)
        elif type(val) is np.ndarray:
            grp.setArray(name, val)
        elif val is None:
            grp.attrs[name] = 'None'
        else:
            # just try saving it as an attr
            try:
                grp.attrs[name] = val
            except Exception, e:
                print 'Warning: Could not save %s, problems may arise in loading.' % name

    group.attrs['__class__'] = obj.__class__.__name__
    for arg in obj._kwargs_init:
        doSave(group, '_kwarg_'+arg, obj._kwargs_init[arg])
    for i, arg in enumerate(obj._args_init):
        doSave(group, '_arg%d'%i, arg)
    obj._savable = group


def loadSavable(node, pointers=None):
    """
        pointers allow things that point to the same node in the h5py file to
        be returned as the same object, if they have already been created.
    """

    if pointers is None: pointers = []
    for pointer in pointers:
        if pointer._savable.node == node.node: return pointer

    args = ([a for a in node.attrs if '_arg' in a] + [a for a in node.children if '_arg' in a])
    kwargs = ([a for a in node.attrs if '_kwarg' in a] + [a for a in node.children if '_kwarg' in a])
    args.sort(key=natural_keys)
    kwargs.sort(key=natural_keys)

    def get(node,key):
        if key in node.children: return node[key]
        elif key in node.attrs: return node.attrs[key]

    ARGS = []
    for name in args:
        val = get(node, name)
        if val.__class__ is h5py.Dataset: val = val[:]
        if val is 'None': val = None
        if '[' in name:  # We are reloading a list
            ind = int(name[4:name.index('[')])
            if len(ARGS) is ind: # Create the list
                ARGS.append([val])
            else:
                ARGS[ind].append(val)
        elif issubclass(val.__class__,hdf5Group):
            ARGS.append(loadSavable(val,pointers=pointers))
        else:
            ind = int(name[4:])
            ARGS.append(val)

    KWARGS = {}
    for name in kwargs:
        val = get(node, name)
        if val.__class__ is h5py.Dataset: val = val[:]
        if val is 'None': val = None
        if '[' in name:  # We are reloading a list
            key = name[7:name.index('[')]
            if key not in KWARGS: # Create the list
                KWARGS[key] = [val]
            else:
                KWARGS[key].append(val)
        elif issubclass(val.__class__,hdf5Group):
            key = name[7:]
            KWARGS[key] = loadSavable(val,pointers=pointers)
        else:
            key = name[7:]
            KWARGS[key] = val

    cls = get(node, '__class__')
    if cls in SAVEABLES:
        try:
            out = SAVEABLES[cls](*ARGS, **KWARGS)
            out._savable = node
            pointers.append(out)  # Because this is recursive.
            return out
        except Exception, e:
            print 'Warning: %s Class could not be initiated.' % cls
            print 'ARGS: ', ARGS
            print 'KWARGS: ', KWARGS
            return (cls, ARGS, KWARGS, node)
    else:
        print 'Warning: %s Class not found in SimPEG.Utils.Save.SAVABLES' % cls
        return (cls, ARGS, KWARGS, node)

