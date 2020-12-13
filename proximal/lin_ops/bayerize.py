from .lin_op import LinOp
import numpy as np
import cv2

import matplotlib.pyplot as plt

class bayerize(LinOp):
    """Samples every steps[i] pixel along axis i,
       starting with pixel 0.
    """

    def __init__(self, arg, pattern='bggr'):
        if pattern != 'bggr':
            raise NotImplementedError()

        if len(arg.shape) != 3:
            raise NotImplementedError()

        self.orig_shape = arg.shape
        shape = (arg.shape[0], arg.shape[1])
        super(bayerize, self).__init__([arg], shape)

    def forward(self, inputs, outputs):
        print("Forward")
        """The forward operator.
        Reads from inputs and writes to outputs.
        """

        a = inputs[0]
        a = (a - a.min())/(a.max()-a.min())
        a = (a * 255).astype(np.uint8)
        cv2.imshow("in", a)
        cv2.waitKey(10)

        # Subsample.
        outputs[0][1::2,1::2] = inputs[0][1::2, 1::2, 0]
        outputs[0][0::2,1::2] = inputs[0][0::2, 1::2, 1]
        outputs[0][1::2,0::2] = inputs[0][1::2, 0::2, 1]
        outputs[0][0::2,0::2] = inputs[0][0::2, 0::2, 2]

    def adjoint(self, inputs, outputs):
        print("Adjoint")
        """The adjoint operator.
        Reads from inputs and writes to outputs.
        """
        # Fill in with zeros.
        outputs[0][:] *= 0

        #np.copyto(outputs[0][1::2, 1::2, 0], inputs[0][1::2, 1::2])

        outputs[0][1::2, 1::2, 0] = inputs[0][1::2, 1::2]
        outputs[0][0::2, 1::2, 1] = inputs[0][0::2, 1::2]
        outputs[0][1::2, 0::2, 1] = inputs[0][1::2, 0::2]
        outputs[0][0::2, 0::2, 2] = inputs[0][0::2, 0::2]

        #print(inputs[0].max())

        #print(outputs[0].max())
        #print(inputs[0][1::2, 1::2].max())

    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        raise NotImplementedError()

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        raise NotImplementedError()

    def is_gram_diag(self, freq=False):
        print("Gram")
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, freq=False):
        print("Get Diag")
        """Returns the diagonal representation (A^TA)^(1/2).
        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert not freq
        var_diags = self.input_nodes[0].get_diag(freq)
        self_diag = np.zeros(self.input_nodes[0].shape)
        
        self_diag[1::2, 1::2, 0] = 1
        self_diag[0::2, 1::2, 1] = 1
        self_diag[1::2, 0::2, 1] = 1
        self_diag[0::2, 0::2, 2] = 1

        self_diag = self_diag.ravel()

        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag
        return var_diags

    def norm_bound(self, input_mags):
        print("Norm bound")
        """Gives an upper bound on the magnitudes of the outputs given inputs.
        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.
        Returns
        -------
        float
            Magnitude of outputs.
        """
        return input_mags[0]