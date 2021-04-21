
import torch
import matplotlib.pyplot as plt

from .utils import get_bounds, plot_input_output

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
        
    def get_rf_coords(self):
        h0, w0 = [self._info[k]['bounds'][0] for k in ['h', 'w']]
        h, w = [self._info[k]['range'] for k in ['h', 'w']]
        return h0, w0, h, w

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
        
        self._info = get_bounds(self.grad_data)

        return self._info

    def info(self):
        return self._info

    def plot(self, fname=None, add_text=False, use_out=None):
        
        plot_input_output(
                          gradient=self.grad_data,
                          output_tensor=self.out,
                          out_pos=self.pos,
                          coords=self.get_rf_coords(),
                          ishape=self.input_shape,
                          fname=fname,
                          add_text=add_text,
                          use_out=use_out
            )

        








