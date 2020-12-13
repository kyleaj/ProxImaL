import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

from proximal import *

input = scipy.misc.face()

input = input[:512, :512, :]

mosaic = np.zeros((input.shape[0], input.shape[1]))
mosaic[1::2,1::2] = input[1::2,1::2, 0]
mosaic[0::2,1::2] = input[0::2,1::2, 1]
mosaic[1::2,0::2] = input[1::2,0::2, 1]
mosaic[0::2,0::2] = input[0::2,0::2, 2]

mosaic = mosaic / 255.0

# Construct and solve problem.
x = Variable(input.shape)



prob = Problem(sum_squares(bayerize(x) - mosaic) + .1 * norm1(grad(x)) + patch_NLM(x))
prob.solve(max_iters=50) #, x0=input/255.0)

# Solve problem.
result = prob.solve(verbose=True, solver='admm')
print('Optimized cost function value = {}'.format(result))

plt.figure(figsize=(15, 8))
plt.subplot(131)
plt.gray()
plt.imshow(input)
plt.title('Original image')

plt.subplot(132)
plt.imshow(mosaic*255, clim=(0, 255))
plt.title('Noisy image')

plt.subplot(133)
plt.imshow(x.value * 255, clim=(0, 255))
plt.title('Denoising results')
plt.show()

print(x.value.min())
print(x.value.max())