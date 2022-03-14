import json

import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import title, tight_layout
from skimage.io import imread, imsave, imshow, show
from matplotlib import pyplot as plt

settings = {'read_path': 'C:\\PyCharmProjects\\mpai_lab1\\images\\12_tank.tif',
            'save_path': 'new.tif'
            }

# open file  and serialize settings in this file
with open('settings.json', 'w') as fp:
    json.dump(settings, fp)

with open('settings.json') as info_data:
    json_data = json.load(info_data)

path = json_data['read_path']
image = imread(path)

# 1) Threshold processing

fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 3, 1)
title('The original image')
imshow(image)

fig.add_subplot(2, 3, 2)
title('The original histogram')
hist, bins = histogram(image)
plt.plot(bins, hist)

# threshold(порог) for image
threshold = 125
threshold_processing_image = image > threshold
threshold_processing_image = (threshold_processing_image * 255).astype(np.uint8)

fig.add_subplot(2, 3, 4)
title('The image after threshold processing')
imshow(threshold_processing_image)

fig.add_subplot(2, 3, 5)
title('The histogram after threshold processing')
hist, bins = histogram(threshold_processing_image)
plt.plot(bins, hist)

fig.add_subplot(2, 3, 6)
# function ravel https://pyprog.pro/array_manipulation/ravel.html
# function ravel return a contiguous flattened array(сжатый до одной оси массив).
x = np.sort(image.ravel())
y = np.sort(threshold_processing_image.ravel())
title('The function element - by - element conversion')
plt.plot(x, y)

show()

# 2) Contrasting
fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 3, 1)
title('The original image')
imshow(image)

fig.add_subplot(2, 3, 2)
title('The original histogram')
hist, bins = histogram(image)
plt.plot(bins, hist)

# [gmin, gmax] - required(acceptable) range
g_min = 0
g_max = 255
# [fmin,fmax] - real range
f_min = image.min()
f_max = image.max()

a = (g_max - g_min) / (f_max - f_min)
b = (g_min * f_max - g_max * f_min) / (f_max - f_min)

contrasted_image = (a * image + b).astype(np.uint8)
fig.add_subplot(2, 3, 4)
title('The contrasted image')
imshow(contrasted_image)

fig.add_subplot(2, 3, 5)
title('The histogram after contrasting processing')
hist, bins = histogram(contrasted_image)
plt.plot(bins, hist)

fig.add_subplot(2, 3, 6)
x = np.sort(image.ravel())
y = np.sort(threshold_processing_image.ravel())
title('The function element - by - element conversion')
plt.plot(x, y)

show()
