import numpy as np
import pandas as pd

def build_log_spaced_layer_thick(first_thk=1, last_dep_top=400, numlay=30, output='thk'):
    '''
    This function builds thickness and depth arrays that have logarithmically increassing thicknesses.
        first_thk: Thickness of first layer
        last_dep_top: depth of the top of the last layer
        numlay: number of layers
        output: default='thk'. Other other is 'DataFrame'
    '''

    roundind = 4

    uni_thk = last_dep_top / numlay
    last_thk = np.round(last_dep_top / numlay,
                        roundind)  # start thickness of uniform layers #potentially save loops in the while loop below
    thk = np.logspace(np.log10(first_thk), np.log10(last_thk), numlay - 1)
    max_dep = np.cumsum(thk)[-1]

    if max_dep > last_dep_top:
        last_thk = first_thk  # if uniform thickness is too big go back and use first thickness
        thk = np.logspace(np.log10(first_thk), np.log10(last_thk), numlay - 1)
        max_dep = np.cumsum(thk)[-1]

    if max_dep > last_dep_top:
        layers = {'error': 'parameters are incompatible and thicknesses cannot increase with depth',
                  'numlay': numlay,
                  'first_thk': first_thk,
                  'last_thk': last_thk,
                  'max_dep': max_dep,
                  'last_dep_top': last_dep_top,
                  }

    else:
        steper = first_thk / 100  # scales based on the input data
        steproundind = len(str(np.float32(steper)).split('.')[1])

        whileind = 0
        last_thk_arr = []
        while max_dep < last_dep_top:
            last_thk += steper
            last_thk = np.round(last_thk, steproundind)
            last_thk_arr.append(last_thk)
            thk = np.logspace(np.log10(first_thk), np.log10(last_thk), numlay - 1)
            dep = np.cumsum(thk)
            max_dep = dep[-1]
            whileind += 1
            del thk, dep

        last_thk = last_thk_arr[whileind - 2]
        thk = np.round(np.logspace(np.log10(first_thk), np.log10(last_thk), numlay - 1), roundind)
        dep_top = np.insert(np.cumsum(thk), 0, 0)
        diff = dep_top[-1] - last_dep_top
        thk[-1] = np.round(thk[-1] - diff, roundind)
        dep_top = np.insert(np.cumsum(thk), 0, 0)
        dep_bot = np.round(np.cumsum(np.append(thk, thk[-1] * 2)), roundind)
        thk = np.append(thk, np.nan)

        roundind = 2

        layers = pd.DataFrame()
        layers['thk'] = np.round(thk, roundind)
        layers['dep_top'] = np.round(dep_top, roundind)
        layers['dep_bot'] = np.round(dep_bot, roundind)

    if output == 'DataFrame':
        return layers
    else:
        return layers.thk.values[:-1]