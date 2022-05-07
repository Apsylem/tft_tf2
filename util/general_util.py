import os
import numpy as np
import pickle


def check_makedir(root_p,folder_list,nested = True):
    """ function to check if a path is already a directory or if it should be created
        takes a list of directorynames to create the path. either nested or on same level.

    Args:
        root_p (path): starting path on which to append directories
        folder_list (str or list of str): name or names of directories to be checked 
        nested (bool, optional): Wether to check/create the directories on the same level, or in a nested manner. Defaults to True.

    Returns:
        [path or list of paths]: path of nested == True, list of paths if nested == False
    """
    
    
    if not isinstance(folder_list,list):
        folder_list=[folder_list]
        
    # if the paths are to be created consecutivly, not on the same level
    # e.g. /rootpath/folder1/folder2/etc...;
    if nested:
        for folder in folder_list:
            path = os.path.join(root_p,folder)
            if not os.path.isdir(path):
                os.mkdir(path)
            root_p = path
        return path
    # if the paths are to be created on 
    # e.g. /rootpath/folder1; /rootpath/folder2, etcs 
    else:
        path_list = []
        for folder in folder_list:
            path = os.path.join(root_p,folder)
            if not os.path.isdir(path):
                os.mkdir(path)
            path_list.append(path)
        return path_list
    
def listdir_fullpath(d):
	return [os.path.join(d, f) for f in os.listdir(d)]

def update_or_init_dict(dict,key,value):
    # checks for an existing key, and update the subdict under this key, if already existing
    if not dict.get(key)==None:
        dict[key].update(value)
    else:
        dict[key]=value
    return dict


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
    

def print_fc(file,*args):
    # prints message in **args to console and already opened file; file need to be closed manually; instead of opening and closing on each call
    if not isinstance(file,bool):
        print(*args,file=file)
    print(*args)
    
def int_False_arg(arg):
    if arg=='False':
        return False
    else:
        return int(arg)
    
def float_False_arg(arg):
    if arg=='False':
        return False
    else:
        return float(arg)

def str_False_arg(arg):
    if arg=='False':
        return False
    else:
        return str(arg)

def str_to_bool(arg):
    if arg=='False':
        return False
    elif arg=='True':
        return True
    

def categorical_or_continus_column(col):
    col = col.replace([np.inf, -np.inf], np.nan).fillna(0)
    if np.all(col.astype(int).astype(np.float32)==col.astype(np.float32)):
        return 'categorical'
    else:
        return 'continues'
    
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def dev_pickle(to_pickle,function_name,abort = True):
    dev_pkl_path = os.path.join('data/dev_pickles',function_name+'.pkl')
    if not os.path.exists(dev_pkl_path):
        if not isinstance(to_pickle,bool):
            pickle.dump(to_pickle, open(dev_pkl_path,'wb'),protocol=4)
            print(f'saved a dev pickle {function_name}')
        else:
            print("somethinhg is to be pickled, that ought not to be. Ima right stop u here")
            print(f'{function_name}')
            print(dev_pkl_path)
        if abort:
            import sys
            sys.exit()
    else:
        
        print(f'loaded a dev pickle {function_name}')
        return pickle.load( open(dev_pkl_path,'rb'))
    
def nan_if_div_by_zero(devident,devisor):
    if not devisor==0:
        return devident/devisor
    else:
        return np.nan
    
def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v
    