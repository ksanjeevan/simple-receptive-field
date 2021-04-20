
import torch
import matplotlib.pyplot as plt

from .utils import get_bounds

class NumericRF:

    def __init__(self, model, input_shape):

        self.model = model.eval()

        if len(input_shape) == 3:
            input_shape = [1] + input_shape

        assert len(input_shape) == 4
        self.input_shape = input_shape


    def _remove_bias(self):
        for conv in self.model:
            conv.bias.data.fill_(0)
            conv.bias.requires_grad = False
        

    def heatmap(self, pos):
        self.pos = pos
        # Step 1: build computational graph
        self.inp = torch.zeros(self.input_shape, requires_grad=True)

        self.out = self.model(self.inp)

        # Step 2: zero out gradient tensor
        grad = torch.zeros_like(self.out, requires_grad=True)

        # Step 3: this could be any non-zero value
        grad[..., pos[0], pos[1]] = 1

        # Step 4: propagate tensor backward
        self.out.backward(gradient=grad)

        # Step 5: average signal over batch and channel + we ony care about magnitute of signal
        self.grad_data = self.inp.grad.mean([0, 1]).abs().data
        
        self.info = get_bounds(self.grad_data)


    def plot(self, fname=None, add_text=False, use_out=None):

        fig = plt.figure(figsize=(13,8))
        ax = [plt.subplot2grid(shape=(4, 1), loc=(0, 0), rowspan=3),
              plt.subplot2grid(shape=(4, 1), loc=(3, 0))]

        # Plot RF
        ishape, oshape = map(lambda x: tuple(x.squeeze().shape), [self.inp, self.out])
        ax[0].set_title("Input shape %s"%(list(ishape)))
        ax[0].imshow(self.grad_data, cmap='copper', interpolation='nearest')

        # Draw RF bounds
        w0, h0, w, h = self.info['w']['bounds'][0], self.info['h']['bounds'][0], self.info['w']['range'], self.info['h']['range']
        ax[0].add_patch(plt.Rectangle((w0-0.5, h0-0.5), w+1, h+1, fill=False, edgecolor='cyan'))

        # Plot channel mean of output
        ax[1].set_title("Output shape %s"%(list(oshape)))
        
        if use_out is not None:
            out = use_out
        else:
            out = self.model(torch.rand(self.input_shape)).detach().mean([0,1]).numpy()

        ax[1].imshow(out, cmap='binary', interpolation='nearest')
        
        ax[1].add_patch(plt.Rectangle((self.pos[1]-0.5, self.pos[0]-0.5), 1, 1, color='cyan'))

        if add_text:
            ax[0].text(w0+w+2, h0, 'Receptive Field', size=17, color='cyan', weight='bold')
            ax[1].text(self.pos[1]+1, self.pos[0], f'{list(self.pos)}', size=19, color='cyan', weight='bold')

        plt.tight_layout()


        if fname is not None:
            plt.savefig(fname, format='png')
            plt.close()









