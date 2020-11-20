import math
import numpy as np


class image:

    def __init__(self):
        self.width = 0
        self.height = 0

    def size(self):
        return (self.width, self.height)

class rgbimage(image):

    def __init__(self):
        image.__init__(self)
        self.r = None
        self.g = None
        self.b = None

    def initfromimread(self, imread):
        self.height, self.width = imread.shape[0:2]
        self.r = imread[:, :, 0]
        self.g = imread[:, :, 1]
        self.b = imread[:, :, 2]

    def initfromyuv(self, yuv):
        self.width, self.height = yuv.size()
        g = yuv.y - ((yuv.u + yuv.v) / 4) # g = y - ((u + v) / 4)
        self.r = np.clip(yuv.v + g, 0, 255) # r = v + g
        self.b = np.clip(yuv.u + g, 0, 255) # b = u + g
        self.g = np.clip(g, 0, 255)
        self.r = self.r.astype(np.uint8)
        self.g = self.g.astype(np.uint8)
        self.b = self.b.astype(np.uint8)

    def getprintable(self):
        printable = np.empty((self.height, self.width, 3), np.uint8)
        for y, row in enumerate(printable):
            for x, pixel in enumerate(row):
                pixel[0] = self.r[y, x]
                pixel[1] = self.g[y, x]
                pixel[2] = self.b[y, x]
        return printable

class yuv(image):

    def __init__(self):
        image.__init__(self)
        self.y = None
        self.u = None
        self.v = None
    
    def initfromimread(self, imread):
        self.height, self.width = imread.shape[0:2]
        imread = imread.astype(np.float64)
        self.y = (imread[:, :, 0] + (2 * imread[:, :, 1]) + imread[:, :, 2]) / 4 # y = (r + 2g + b) / 4
        self.u = imread[:, :, 2] - imread[:, :, 1] # u = b - g
        self.v = imread[:, :, 0] - imread[:, :, 1] # v = r - g

    def initfromrgbimage(self, rgbimage):
        self.width, self.height = rgbimage.size()
        rgbimage.r = rgbimage.r.astype(np.float64)
        rgbimage.g = rgbimage.g.astype(np.float64)
        rgbimage.b = rgbimage.b.astype(np.float64)
        self.y = (rgbimage.r + (2 * rgbimage.g) + rgbimage.b) / 4 # y = (r + 2g + b) / 4
        self.u = rgbimage.b - rgbimage.g # u = b - g
        self.v = rgbimage.r - rgbimage.g # v = r - g
    
    def initfromsubsample(self, subsample):
        self.y = subsample.y.copy()
        self.u = np.empty(self.y.shape)
        self.v = np.empty(self.y.shape)
        self.width, self.height = subsample.size()

        if subsample.subsampling == (4, 2, 0):
            self.u[::2, ::2] = subsample.u[:, :]
            self.u[::2, 1::2] = subsample.u[:, :]
            self.u[1::2, ::2] = subsample.u[:, :]
            self.u[1::2, 1::2] = subsample.u[:, :]
            self.v[::2, ::2] = subsample.v[:, :]
            self.v[::2, 1::2] = subsample.v[:, :]
            self.v[1::2, ::2] = subsample.v[:, :]
            self.v[1::2, 1::2] = subsample.v[:, :]

        elif subsample.subsampling == (4, 2, 2):
            self.u[:, ::2] = subsample.u[:, :]
            self.u[:, 1::2] = subsample.u[:, :]
            self.v[:, ::2] = subsample.v[:, :]
            self.v[:, 1::2] = subsample.v[:, :]
        elif subsample.subsampling == (4, 4, 4):
            self.u = subsample.u.copy()
            self.v = subsample.v.copy()

class subsample(image):

    _supportedsubsampling = [(4, 2, 0), (4, 2, 2), (4, 4, 4)]
    def __init__(self, subsampling: tuple = (4, 2, 0)):
        if subsampling not in subsample._supportedsubsampling:
            raise ValueError("Unsupported subsampling", subsampling, "supported subsampling: ", subsample._supportedsubsampling)

        image.__init__(self)
        self.subsampling = subsampling
        self.y = None
        self.u = None
        self.v = None

    def initfromyuv(self, yuv):
        self.width, self.height = yuv.size()
        self.y = yuv.y
        
        if self.subsampling == (4, 2, 0):
            self.u = yuv.u[::2, ::2] # keep 1 row in 2 and 1 element in 2 from each row
            self.v = yuv.v[::2, ::2]
        elif self.subsampling == (4, 2, 2):
            self.u = yuv.u[:, ::2] # keep every row and 1 element in 2 from each row
            self.v = yuv.v[:, ::2]
        elif self.subsampling == (4, 4, 4):
            self.u = yuv.u
            self.v = yuv.v
        else:
            print("This was not supposed to happen...")
    
    def initfromdwt(self, dwt, subsampling):
        originalwidth, originalheight = dwt.size()
        ywidth, yheight, uwidth, uheight, vwidth, vheight = subsample._getshape(originalwidth, originalheight, dwt.recursionlevel - 1, subsampling)
        ylxly = dwt.y.reshape(yheight, ywidth)
        ulxly = dwt.u.reshape(uheight, uwidth)
        vlxly = dwt.v.reshape(vheight, vwidth)
        
        for i in reversed(range(dwt.recursionlevel)):
            ywidth, yheight, uwidth, uheight, vwidth, vheight = subsample._getshape(originalwidth, originalheight, i, subsampling)
            ylxly = subsample._getdwtoriginal(
                ylxly,
                dwt.reconstructiondata[i]["ylxhy"].reshape((yheight, ywidth)),
                dwt.reconstructiondata[i]["yhxly"].reshape((yheight, ywidth)),
                dwt.reconstructiondata[i]["yhxhy"].reshape((yheight, ywidth))
            )
            ulxly = subsample._getdwtoriginal(
                ulxly,
                dwt.reconstructiondata[i]["ulxhy"].reshape((uheight, uwidth)),
                dwt.reconstructiondata[i]["uhxly"].reshape((uheight, uwidth)),
                dwt.reconstructiondata[i]["uhxhy"].reshape((uheight, uwidth))
            )
            vlxly = subsample._getdwtoriginal(
                vlxly,
                dwt.reconstructiondata[i]["vlxhy"].reshape((vheight, vwidth)),
                dwt.reconstructiondata[i]["vhxly"].reshape((vheight, vwidth)),
                dwt.reconstructiondata[i]["vhxhy"].reshape((vheight, vwidth))
            )
        
        self.y = ylxly
        self.u = ulxly
        self.v = vlxly
        self.subsampling = subsampling
        self.width, self.height = dwt.size()
    
    @staticmethod
    def _getshape(originalwidth, originalheight, recursionlevel, subsampling):
        ywidth = originalwidth / (2**(recursionlevel + 1))
        yheight = originalheight / (2**(recursionlevel + 1))
        
        if subsampling == (4, 2, 0):
            uwidth = vwidth = ywidth / 2
            uheight = vheight = yheight / 2
        elif subsampling == (4, 2, 2):
            uwidth = vwidth = ywidth / 2
            uheight = vheight = yheight
        elif subsampling == (4, 4, 4):
            uwidth = vwidth = ywidth
            uheight = vheight = yheight
        else:
            print("whyyyyy?")
        
        return int(ywidth), int(yheight), int(uwidth), int(uheight), int(vwidth), int(vheight)
    
    @staticmethod
    def _getdwtoriginal(lxly, lxhy, hxly, hxhy):
        height, width = lxly.shape
        # get lx
        lx = np.empty((2 * height, width))
        lx[::2, :] = lxly[:, :] + lxhy[:, :]
        lx[1::2, :] = lxly[:, :] - lxhy[:, :]
        #get hx
        hx = np.empty((2 * height, width))
        hx[::2, :] = hxly[:, :] + hxhy[:, :]
        hx[1::2, :] = hxly[:, :] - hxhy[:, :]
        #get original
        height, width = lx.shape
        original = np.empty((height, 2 * width))
        original[:, ::2] = lx[:, :] + hx[:, :]
        original[:, 1::2] = lx[:, :] - hx[:, :]

        return original

class dwt(image):

    def __init__(self):
        image.__init__(self)
        self.recursionlevel = 1
        self.y = None
        self.u = None
        self.v = None
        self.reconstructiondata = []

    def initfromsubsample(self, subsample, recursionlevel):
        self.width, self.height = subsample.size()
        self.y = subsample.y.copy()
        self.u = subsample.u.copy()
        self.v = subsample.v.copy()
        self.recursionlevel = recursionlevel

        for _ in range(recursionlevel):
            ylx = dwt._filter(self.y, "lowpass", 'x')
            ulx = dwt._filter(self.u, "lowpass", 'x')
            vlx = dwt._filter(self.v, "lowpass", 'x')

            yhx = dwt._filter(self.y, "highpass", 'x')
            uhx = dwt._filter(self.u, "highpass", 'x')
            vhx = dwt._filter(self.v, "highpass", 'x')

            ylxly = dwt._filter(ylx, "lowpass", 'y')
            ulxly = dwt._filter(ulx, "lowpass", 'y')
            vlxly = dwt._filter(vlx, "lowpass", 'y')

            ylxhy = dwt._filter(ylx, "highpass", 'y')
            ulxhy = dwt._filter(ulx, "highpass", 'y')
            vlxhy = dwt._filter(vlx, "highpass", 'y')

            yhxly = dwt._filter(yhx, "lowpass", 'y')
            uhxly = dwt._filter(uhx, "lowpass", 'y')
            vhxly = dwt._filter(vhx, "lowpass", 'y')

            yhxhy = dwt._filter(yhx, "highpass", 'y')
            uhxhy = dwt._filter(uhx, "highpass", 'y')
            vhxhy = dwt._filter(vhx, "highpass", 'y')

            self.y = ylxly
            self.u = ulxly
            self.v = vlxly
            self.reconstructiondata.append({
                "ylxhy": ylxhy,
                "yhxly": yhxly,
                "yhxhy": yhxhy,
                "ulxhy": ulxhy,
                "uhxly": uhxly,
                "uhxhy": uhxhy,
                "vlxhy": vlxhy,
                "vhxly": vhxly,
                "vhxhy": vhxhy
            })
    
    def initfromquantize(self, quantize):
        self.recursionlevel = quantize.recursionlevel
        self.y = dwt._dequantize(quantize.y, quantize.deadzone, quantize.step)
        self.u = dwt._dequantize(quantize.u, quantize.deadzone, quantize.step)
        self.v = dwt._dequantize(quantize.v, quantize.deadzone, quantize.step)
        self.reconstructiondata = []
        self.width, self.height = quantize.size()
        self.recursionlevel = quantize.recursionlevel
        
        for i in quantize.reconstructiondata:
            self.reconstructiondata.append({
                "ylxhy": dwt._dequantize(i["ylxhy"], quantize.deadzone, quantize.step),
                "yhxly": dwt._dequantize(i["yhxly"], quantize.deadzone, quantize.step),
                "yhxhy": dwt._dequantize(i["yhxhy"], quantize.deadzone, quantize.step),
                "ulxhy": dwt._dequantize(i["ulxhy"], quantize.deadzone, quantize.step),
                "uhxly": dwt._dequantize(i["uhxly"], quantize.deadzone, quantize.step),
                "uhxhy": dwt._dequantize(i["uhxhy"], quantize.deadzone, quantize.step),
                "vlxhy": dwt._dequantize(i["vlxhy"], quantize.deadzone, quantize.step),
                "vhxly": dwt._dequantize(i["vhxly"], quantize.deadzone, quantize.step),
                "vhxhy": dwt._dequantize(i["vhxhy"], quantize.deadzone, quantize.step)
            })

    @staticmethod
    def _filter(channel, ftype, axis):
        if ftype not in ["lowpass", "highpass"]:
            raise ValueError("Invalid filter type", ftype, "valid filter types: ", ["lowpass", "highpass"])
        if axis not in ['x', 'y']:
            raise ValueError("Invalid axis", axis, "valid axis: ", ['x', 'y'])

        if axis == 'x':
            evens = channel[:, ::2] # all rows, 1 in 2 elements starting from element 0
            odds = channel[:, 1::2] # all rows, 1 in 2 elements starting from element 1
            if evens.shape == odds.shape:
                if ftype == "lowpass":
                    return (evens + odds) / 2
                elif ftype == "highpass":
                    return (evens - odds) / 2
            else: # if shapes not equal, evens will always have 1 more element in its rows
                lastelements = evens[:, -1:]
                allexceptlastelements = evens[:, :-1]
                if ftype == "lowpass":
                    result = (allexceptlastelements + odds) / 2
                elif ftype == "highpass":
                    result = (allexceptlastelements - odds) / 2
                return np.concatenate((result, lastelements), axis = 1)
        elif axis == 'y':
            evens = channel[::2, :] # 1 row in 2 starting from row 0, all elements
            odds = channel[1::2, :] # 1 row in 2 starting from row 1, all elements
            if evens.shape == odds.shape:
                if ftype == "lowpass":
                    return (evens + odds) / 2
                elif ftype == "highpass":
                    return (evens - odds) / 2
            else: # if shapes not equal, evens will always have 1 row
                lastrow = evens[-1:, :]
                allexceptlastrow = evens[:-1, :]
                if ftype == "lowpass":
                    result = (allexceptlastrow + odds) / 2
                elif ftype == "highpass":
                    result = (allexceptlastrow - odds) / 2
                return np.concatenate((result, lastrow), axis = 0)
    
    @staticmethod
    def _dequantize(vector, deadzone, step):
        return np.array([np.sign(x) * ((deadzone / 2) + (step * (abs(x) - 1 + 0.5))) for x in vector])

class quantize(image):

    def __init__(self):
        image.__init__(self)
        self.y = None
        self.u = None
        self.v = None
        self.reconstructiondata = []
        self.deadzone = 0
        self.step = 1
        self.recursionlevel = 1
    
    def initfromdwt(self, dwt, deadzone, step):
        self.width, self.height = dwt.size()
        self.y = quantize._quantize(dwt.y.flatten(), deadzone, step)
        self.u = quantize._quantize(dwt.u.flatten(), deadzone, step)
        self.v = quantize._quantize(dwt.v.flatten(), deadzone, step)
        self.deadzone = deadzone
        self.step = step
        self.recursionlevel = dwt.recursionlevel

        for i in dwt.reconstructiondata:
            self.reconstructiondata.append({
                "ylxhy": quantize._quantize(i["ylxhy"].flatten(), deadzone, step),
                "yhxly": quantize._quantize(i["yhxly"].flatten(), deadzone, step),
                "yhxhy": quantize._quantize(i["yhxhy"].flatten(), deadzone, step),
                "ulxhy": quantize._quantize(i["ulxhy"].flatten(), deadzone, step),
                "uhxly": quantize._quantize(i["uhxly"].flatten(), deadzone, step),
                "uhxhy": quantize._quantize(i["uhxhy"].flatten(), deadzone, step),
                "vlxhy": quantize._quantize(i["vlxhy"].flatten(), deadzone, step),
                "vhxly": quantize._quantize(i["vhxly"].flatten(), deadzone, step),
                "vhxhy": quantize._quantize(i["vhxhy"].flatten(), deadzone, step)
            })
    
    @staticmethod
    def _quantize(vector, deadzone, step):
        return np.array([np.sign(x) * max(0, math.floor(((abs(x) - (deadzone / 2)) / step) + 1)) for x in vector])

class lzw(image):

    def __init__(self):
        image.__init__(self)
        self.y = None
        self.ydict = None
        self.u = None
        self.udict = None
        self.v = None
        self.vdict = None
        self.reconstructiondata = []
    
    def initfromquantize(self, quantize):
        self.width, self.height = quantize.size()
        self.ydict, self.y = lzw._encode(quantize.y)
        self.udict, self.u = lzw._encode(quantize.u)
        self.vdict, self.v = lzw._encode(quantize.v)

        for i in quantize.reconstructiondata:
            ylxhydict, ylxhy = lzw._encode(i["ylxhy"])
            yhxlydict, yhxly = lzw._encode(i["yhxly"])
            yhxhydict, yhxhy = lzw._encode(i["yhxhy"])
            ulxhydict, ulxhy = lzw._encode(i["ulxhy"])
            uhxlydict, uhxly = lzw._encode(i["uhxly"])
            uhxhydict, uhxhy = lzw._encode(i["uhxhy"])
            vlxhydict, vlxhy = lzw._encode(i["vlxhy"])
            vhxlydict, vhxly = lzw._encode(i["vhxly"])
            vhxhydict, vhxhy = lzw._encode(i["vhxhy"])
            self.reconstructiondata.append({
                "ylxhy": ylxhy,
                "ylxhydict": ylxhydict,
                "yhxly": yhxly,
                "yhxlydict": yhxlydict,
                "yhxhy": yhxhy,
                "yhxhydict": yhxhydict,
                "ulxhy": ulxhy,
                "ulxhydict": ulxhydict,
                "uhxly": uhxly,
                "uhxlydict": uhxlydict,
                "uhxhy": uhxhy,
                "uhxhydict": uhxhydict,
                "vlxhy": vlxhy,
                "vlxhydict": vlxhydict,
                "vhxly": vhxly,
                "vhxlydict": vhxlydict,
                "vhxhy": vhxhy,
                "vhxhydict": vhxhydict,
            })
    
    @staticmethod
    def _getinitdict(vector):
        symbols = np.array([], dtype = int)
        
        for symb in vector:
            if int(symb) not in symbols:
                symbols = np.append(symbols, int(symb))
        
        initdict = {}
        symbols.sort()
        for i in range(symbols.size):
            initdict[str(symbols[i])] = "{:b}".format(i).zfill(int(np.ceil(np.log2(symbols.size))))

        return initdict
    
    @staticmethod
    def _encode(vector):
        initdict = lzw._getinitdict(vector)
        encdict = initdict.copy()
        encoded = np.array([], dtype = str)
        pos = 0

        while pos < vector.size:
            subsymbols = str(int(vector[pos]))
            subsymbolsinencdict = str(int(vector[pos]))

            while subsymbols in encdict and pos < vector.size:
                subsymbolsinencdict = subsymbols
                pos += 1
                if pos < vector.size:
                    subsymbols += str(int(vector[pos]))

            encoded = np.append(encoded, encdict[subsymbolsinencdict])

            if pos < vector.size:
                encdict[subsymbols] = "{:b}".format(len(encdict))

                if np.ceil(np.log2(len(encdict))) > len(encoded[-1]):
                    for symb, code in encdict.items():
                        encdict[symb] = code.zfill(int(np.ceil(np.log2(len(encdict)))))
        
        return initdict, encoded
    
    @staticmethod
    def getsize(initdict, encoded):
        size = len(initdict) * len(initdict[(next(iter(initdict)))]) + (len(initdict) * 9) # initdict size in bits

        for code in encoded:
            size += len(code)
        
        return size

class compressedimage(image):
    def __init__(self, imread, yuvsubsamp = (4, 2, 0), dwtrecurslevel = 3, quantizdeadzone = 4, quantizstep = 1):
        print("subsampling:\t\t", yuvsubsamp)
        print("dwt recursions:\t\t", dwtrecurslevel)
        print("quantizer deadzone:\t", quantizdeadzone)
        print("quantizer step:\t\t", quantizstep)

        image.__init__(self)
        self.rgbimage = rgbimage()
        self.yuv = yuv()
        self.subsample = subsample(yuvsubsamp)
        self.dwt = dwt()
        self.quantize = quantize()
        self.lzw = lzw()

        self.yuv.initfromimread(imread)
        self.width, self.height = self.yuv.size()
        self.subsample.initfromyuv(self.yuv)
        self.dwt.initfromsubsample(self.subsample, dwtrecurslevel)
        self.quantize.initfromdwt(self.dwt, quantizdeadzone, quantizstep)
        self.lzw.initfromquantize(self.quantize)

    def getprintable(self):
        tmpdwt = dwt()
        tmpdwt.initfromquantize(self.quantize)
        tmpsubsample = subsample()
        tmpsubsample.initfromdwt(tmpdwt, self.subsample.subsampling)
        tmpyuv = yuv()
        tmpyuv.initfromsubsample(tmpsubsample)
        tmprgbimage = rgbimage()
        tmprgbimage.initfromyuv(tmpyuv)
        printable = tmprgbimage.getprintable()
        return printable
    
    def getuncompressedsize(self):
        return self.width * self.height * 3 * 8 / 1000 # in kilobits
    
    def getcompressedsize(self):
        size = 0
        size += lzw.getsize(self.lzw.ydict, self.lzw.y)
        size += lzw.getsize(self.lzw.udict, self.lzw.u)
        size += lzw.getsize(self.lzw.vdict, self.lzw.v)

        for i in self.lzw.reconstructiondata:
            size += lzw.getsize(i["ylxhydict"], i["ylxhy"])
            size += lzw.getsize(i["yhxlydict"], i["yhxly"])
            size += lzw.getsize(i["yhxhydict"], i["yhxhy"])
            size += lzw.getsize(i["ulxhydict"], i["ulxhy"])
            size += lzw.getsize(i["uhxlydict"], i["uhxly"])
            size += lzw.getsize(i["uhxhydict"], i["uhxhy"])
            size += lzw.getsize(i["vlxhydict"], i["vlxhy"])
            size += lzw.getsize(i["vhxlydict"], i["vhxly"])
            size += lzw.getsize(i["vhxhydict"], i["vhxhy"])
        
        return size / 1000 # in kilobits