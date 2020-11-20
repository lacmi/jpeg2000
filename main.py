import numpy as np
import matplotlib.pyplot as py

import libinf8770 as mylib


imread = py.imread("power.jpg")

py.figure()
py.imshow(imread)

compressedimage = mylib.compressedimage(imread, yuvsubsamp = (4,2,0), dwtrecurslevel = 3, quantizdeadzone = 2, quantizstep = 2)

uncompressedsize = compressedimage.getuncompressedsize()
compressedsize = compressedimage.getcompressedsize()

print("uncompressed size:\t", uncompressedsize, " kb")
print("compressed size:\t", compressedsize, " kb")
print("compression rate:\t{0:1.4f}".format(1 - (compressedsize / uncompressedsize)))

printable = compressedimage.getprintable()
py.figure()
py.imshow(printable)

py.show()