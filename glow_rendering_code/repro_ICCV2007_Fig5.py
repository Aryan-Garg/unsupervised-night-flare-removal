#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from skimage import io, filters

def generate_APSF(q, T, k):
    p = k * T
    sigma = (1 - q) / q
    mu = 0

    A = np.sqrt(sigma ** 2 * gamma(1 / p) / gamma(3 / p))

    x = np.linspace(-6, 6, 100)
    XX, YY = np.meshgrid(x, x)
    APSF2D = np.exp(-((XX ** 2 + YY ** 2) ** (p / 2)) / abs(A) ** p) / (2 * gamma(1 + 1 / p) * A) ** 2
    APSF2D = APSF2D / np.sum(APSF2D)

    return APSF2D

clean_path = './img/ICCV2007/0009.png'
clean_img = io.imread(clean_path).astype(float) / 255.0

q_cand = [0.70, 0.55, 0.3]
T = 1.2
k = 0.5

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(clean_img)
axes[0, 0].set_title('Clean')

x = np.linspace(-6, 6, 100)

for i, q in enumerate(q_cand):
    APSF2D = generate_APSF(q, T, k)
    img = filters.convolve(clean_img, APSF2D)

    axes[0, i + 1].imshow(img)
    axes[0, i + 1].set_title(f'T={T:.1f}, q={q:.2f}')

    axes[1, i + 1].plot(x, APSF2D[:, 50] / np.sum(APSF2D), linewidth=1.5)

plt.show()
