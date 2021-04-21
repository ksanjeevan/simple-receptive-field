
import numpy as np
import matplotlib.pyplot as plt

def get_bounds(gradient):
    
    coords = gradient.nonzero(as_tuple=True) # get non-zero coords
    names = ['h', 'w']
    ret = {}
    
    for ind in [0, 1]:
        mini, maxi = coords[ind].min().item(), coords[ind].max().item()
        ret[names[ind]] = {'bounds' : (mini, maxi), 'range': maxi - mini}
    return ret


def plot_input_output(gradient, output_tensor, out_pos, coords, ishape, fname=None, add_text=False, use_out=None):
    fig = plt.figure(figsize=(13,8))
    ax = [plt.subplot2grid(shape=(4, 1), loc=(0, 0), rowspan=3),
    plt.subplot2grid(shape=(4, 1), loc=(3, 0))]

    # Plot RF
    oshape = output_tensor.squeeze().shape

    ax[0].set_title("Input shape %s"%(list(ishape)))
    ax[0].imshow(gradient, cmap='copper', interpolation='nearest')

    # Draw RF bounds
    h0, w0, h, w = coords
    ax[0].add_patch(plt.Rectangle((w0-0.5, h0-0.5), w+1, h+1, fill=False, edgecolor='cyan'))

    # Plot channel mean of output
    ax[1].set_title("Output shape %s"%(list(oshape)))

    if use_out is not None:
        out = use_out
    else:

        out = np.random.rand(*oshape)

    ax[1].imshow(out, cmap='binary', interpolation='nearest')

    ax[1].add_patch(plt.Rectangle((out_pos[1]-0.5, out_pos[0]-0.5), 1, 1, color='cyan'))

    if add_text:
        ax[0].text(w0+w+2, h0, 'Receptive Field', size=17, color='cyan', weight='bold')
        ax[1].text(out_pos[1]+1, out_pos[0], f'{list(out_pos)}', size=19, color='cyan', weight='bold')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, format='png')
        plt.close()
