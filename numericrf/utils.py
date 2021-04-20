


def get_bounds(gradient):
    
    coords = gradient.nonzero(as_tuple=True) # get non-zero coords
    names = ['h', 'w']
    ret = {}
    
    for ind in [0, 1]:
        mini, maxi = coords[ind].min().item(), coords[ind].max().item()
        ret[names[ind]] = {'bounds' : (mini, maxi), 'range': maxi - mini}
    return ret
