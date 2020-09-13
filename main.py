import time
import tkinter as tk
from pprint import pprint
from threading import Thread
from tkinter.ttk import *
import numpy as np
import pywt
from PIL import Image, ImageFilter
import cv2
from tkinter.filedialog import askopenfilename
from PIL import ImageTk
from PIL.ImageColor import colormap
from matplotlib import pyplot as plt, cm  # import pylot
from matplotlib.colors import Normalize
from matplotlib.pyplot import ginput
from numpy import size
from scipy._lib.six import xrange
from scipy.constants import point
from scipy.ndimage import interpolation
from skimage import filters, color, feature
from skimage.transform import radon, iradon
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import datasets
import sklearn.metrics as sm

pencere = tk.Tk()
pencere.title("Görüntü İşleme")
pencere.geometry("700x600")
pencere.configure(background="thistle")
pencere.resizable(width=False, height=False)

# Fonksiyonlar
def resimac():
    global tk_i, name, i, width, height, i

    name = askopenfilename(initialdir="/", title="Dosya Seç", filetypes=(("jpeg", "*.jpg"), ("all files", "*.*")))
    i = Image.open(name)
    width = 300
    height = 400
    i2 = i.resize((width, height), Image.ANTIALIAS)
    tk_i = ImageTk.PhotoImage(i2)

    label = tk.Label(pencere, image=tk_i, anchor=tk.W)
    label.place(x=200, y=80)

    # a = canvas.create_image(5, 5, image=tk_i, anchor=tk.NW)
    print(i)


def dondur():
    global img, a, b
    a = cv2.imread(name, 0)
    rows, cols = a.shape
    #if combobox.get() == "90":
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(a, M, (cols, rows))
    cv2.imshow('90 derece', dst)
    #if combobox.get() == "180":
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst = cv2.warpAffine(a, M, (cols, rows))
    cv2.imshow('180 derece', dst)
    #if combobox.get() == "270":
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
    dst = cv2.warpAffine(a, M, (cols, rows))
    cv2.imshow('270 derece', dst)


def gri():
    global cevir, im_gri, a
    im_gri = cv2.imread(name)
    cevir = cv2.cvtColor(im_gri, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Gri", cevir)

# Histogram Eşitleme
def histogram():
    img = cv2.imread(name, 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imshow("histogram", res)


def esikleme():
    img = cv2.imread(name)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['ORJINAL', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in xrange(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def otsu():
    img = cv2.imread(name, 0)
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in xrange(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


# Morfolojik İşlemler
def asinma():
    img = cv2.imread(name, 0)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imshow("erozyon", erosion)


def genisleme():
    img = cv2.imread(name, 0)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow("dilation", dilation)


def acma():
    img = cv2.imread(name, 0)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Açma", opening)


def kapama():
    img = cv2.imread(name, 0)
    kernel = np.ones((4, 4), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Kapama", closing)


def gradyan():
    img = cv2.imread(name)
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Gradyan", gradient)


# Hough Dönüşümü
def daire():
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """"output=img.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.imshow("output", np.hstack([img, output]))"""""

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32,
                               param1=40, param2=10, minRadius=5, maxRadius=10)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    plt.subplot(121), plt.imshow(gray)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img)
    plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
    plt.show()

def cizgi():
    global r, theta
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=250)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
    """for r, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)"""
    cv2.imshow("Edges", edges)
    cv2.imshow("Çizgi", img)

# Filtreler

#def filtreler():
    #if combo.get() == "Median":
def median():
        im = cv2.imread(name, 0)
        median = cv2.medianBlur(im, 5)
        compare = np.concatenate((im, median), axis=1)
        cv2.imshow("MedianBlur", compare)

def mean():
        im = cv2.imread(name)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        """"def filterImage(image, kernel):

            ret = image.copy()
            y = image.shape[0]
            x = image.shape[1]
            kernel_size = len(kernel)
            a = kernel_size // 2
            for i in range(a, y - a):
                for j in range(a, x - a):
                    val = 0
                    for k in range(kernel_size * kernel_size):
                        im_val = image[i + k // kernel_size - a][j + k % kernel_size - a]
                        val += kernel[k // kernel_size][k % kernel_size] * im_val
                    ret[i][j] = val
            return ret

        kernel_s = 5  # averaging kernel
        kernel = np.ones([kernel_s ** 2])
        kernel = kernel // np.sum(kernel)
        kernel = np.reshape(kernel, [kernel_s, kernel_s])

        im1_mean = filterImage(im, kernel)
        fig = plt.figure()
        plt.subplot('121'), plt.imshow(im, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot('122'), plt.imshow(im1_mean, cmap='gray')
        plt.title('Mean Image'), plt.xticks([]), plt.yticks([])
        fig.show()"""""
        kernel = np.ones((5, 5), np.float32) / 15
        dst = cv2.filter2D(img, -1, kernel)

        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Mean')
        plt.xticks([]), plt.yticks([])
        plt.show()

def gaussian():
        im = cv2.imread(name)
        blur = cv2.GaussianBlur(im, (5, 5), 0)
        cv2.imshow("Gaussian", blur)

        """def gaussian_kernel(size,size_y=None):
            size = int(size)
            if not size_y:
                size_y = im.size
            else:
                size_y = int(size_y)
            x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
            g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
            return g / g.sum()
        gaussian_kernel_array = gaussian_kernel(5)
        plt.imshow(gaussian_kernel_array,
         cmap=plt.get_cmap('jet'), interpolation='nearest')
        plt.colorbar()
        plt.show()"""

def laplacian():
        im = cv2.imread(name,0)
        laplacian = cv2.Laplacian(im, cv2.CV_64F)
        cv2.imshow("Laplacian", laplacian)

def sobel():
        img = cv2.imread(name, 0)
        sobel_dikey = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel_yatay = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        plt.subplot(121), plt.imshow(sobel_yatay, cmap='gray')
        plt.title('Sobel Yatay'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(sobel_dikey, cmap='gray')
        plt.title('Sobel Dikey'), plt.xticks([]), plt.yticks([])
        plt.show()
        # Convolvee
        """height = img.shape[0]
        width = img.shape[1]

        Hx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        Hy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        t0 = time.time()

        img_x = convolve_np(img, Hx) / 8.0
        img_y = convolve_np(img, Hy) / 8.0

        img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

        img_out = (img_out / np.max(img_out)) * 255

        t1 = time.time()
        print(t1 - t0)

        cv2.imwrite('images/edge_sobel.jpg', img_out)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()"""

def prewitt():
        im = cv2.imread(name)
        im1 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(im1, (3, 3), 0)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img, -1, kernelx)
        img_prewitty = cv2.filter2D(img, -1, kernely)
        plt.subplot(121), plt.imshow(img_prewittx, cmap='gray')
        plt.title('Prewitt Yatay'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_prewitty, cmap='gray')
        plt.title('Prewitt Dikey'), plt.xticks([]), plt.yticks([])
        plt.show()

        # edge = filters.prewitt(im)
        # cv2.imshow("prewit", edge)

        # Convolve
        """im = cv2.imread(name)
        height = im.shape[0]
        width = im.shape[1]

        Hy = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

        Hx = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

        im_x = convolve_np(im, Hx) / 6.0
        im_y = convolve_np(im, Hy) / 6.0

        im_out = np.sqrt(np.power(im_x, 2) + np.power(im_y, 2))

        im_out = (im_out / np.max(im_out)) * 255

        cv2.imwrite('edge/prewit', im_out)
        plt.imshow(im_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()


# X and F are numpy matrices
def convolve_np(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height - 1) / 2
    W = (F_width - 1) / 2

    out = np.zeros((X_height, X_width))

    for i in np.arange(H, X_height - H):
        for j in np.arange(W, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum

    return out"""

def min():
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img_out = img.copy()
        height = img.shape[0],width = img.shape[1]
        for i in np.arange(3, height - 3):
            for j in np.arange(3, width - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                b = min
                img_out.itemset((i, j), b)
                cv2.imwrite('minfilter.jpg', img_out)
                cv2.imshow("minfilter", img_out)
def max():
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img_out = img.copy()
        height = img.shape[0], width = img.shape[1]
        for i in np.arange(3, height - 3):
            for j in np.arange(3, width - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                img_out.itemset((i, j), b)
                cv2.imwrite('maxfilter.jpg', img_out)
                cv2.imshow('max', img_out)
        
def kumeleme():
    """dataset = cv2.imread(name)
    X = dataset.shape[:, [3, 4]]
    wcss = []
    kume_sayisi_listesi = range(1, 11)
    for i in kume_sayisi_listesi:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        plt.plot(kume_sayisi_listesi, wcss)
        plt.title('Küme Sayısı Belirlemek İçin Dirsek Yöntemi')
        plt.xlabel('Küme Sayısı')
        plt.ylabel('WCSS')
        plt.show()

        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_means = kmeans.fit_predict(X)

        plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=100, c='red', label='Küme 1')
        plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=100, c='blue', label='Küme 2')
        plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=100, c='green', label='Küme 3')
        plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=100, c='cyan', label='Küme 4')
        plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=100, c='magenta', label='Küme 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow',
                    label='Küme Merkezleri')
        plt.title('Müşteri Segmentasyonu')
        plt.xlabel('Yıllık Gelir')
        plt.ylabel('Harcama Skoru (1-100)')
        plt.legend()
        plt.show()"""

    im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    Z = im.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv2.kmeans(Z, K, None,
                                    criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((im.shape))
    cv2.imshow('Kmeans', res2)

    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]

    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()

    """iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    x.columns = ['Image_Length', 'Image_Width']

    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    plt.figure(figsize=(14, 7))
    colormap = np.array(['red', 'blue', 'black'])

    plt.subplot(1, 2, 1)
    plt.scatter(x.Image_Length, x.Image_Width, c=colormap[y.Targets], s=40)
    plt.title('Sepal')

    model = KMeans(n_clusters=3)
    model.fit(x)

    predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

    plt.figure(figsize=(14,7))
    colormap=np.array(['red','blue','black'])

    plt.subplot(1,2,1)
    plt.scatter(x.Image_Length, x.Image_Width, c=colormap[y.Targets], s=40)
    plt.title('Gerçek Sınıflandırma')

    plt.subplot(1,2,2)
    plt.scatter(x.Image_Length, x.Image_Width, c=colormap[predY], s=40)
    plt.title('K-Means Sınıflandırması')
    print(sm.accuracy_score(y, predY))"""

def karsitlik(min=0, max=255):
    A = cv2.imread(name, 0)
    x,y = A.shape
    ret= A.copy()
    for i in range(x):
        for j in range(y):
            ret[i, j] = ((A[i,j] - min) / (max-min)) *255

    cv2.imshow("Karşıtlık", ret)

def DCT():
        im1 = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        imf = np.float32(im1) / 255.0  # float conversion/scale
        dst = cv2.dct(imf)  # the dct
        #img = np.uint8(dst) * 255.0  # convert back
        cv2.imshow("DCT", dst)
        """h, w = np.array(im1.shape[:2]) // B ** B
        print(h)
        print(w)
        im1 = im1[:h, :w]

        sV = h // B
        sH = w // B
        vis = np.zeros((h, w), dtype=np.float32)
        trans = np.zeros((h, w), dtype=np.float32)
        vis[:h, :w] = im1
        for row in range(sV):
            for col in range(sH):
                currentblock = cv2.dct(vis[row * B:(row + 1) * B, col * B:(col + 1) * B])
                trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock

        # cv2.imshow("Resim", im1)
        plt.imshow(im1, cmap="gray")
        point = [(10, 10), (18, 18)]
        block = np.floor(np.array(point) // B)
        col = block[0, 0]
        row = block[0, 1]
        plt.plot([B * col, B * col + B, B * col + B, B * col, B * col],
                 [B * row, B * row, B * row + B, B * row + B, B * row])
        plt.axis([0, w, h, 0])
        plt.title("Orjinal resim")

        plt.figure()
        plt.subplot(1, 2, 1)
        selectedImg = im1[row * B:(row + 1) * B, col * B:(col + 1) * B]
        N255 = Normalize(0, 255)
        plt.imshow(selectedImg, cmap="gray", norm=N255, interpolation='nearest')
        plt.title("Image")

        plt.subplot(1, 2, 2)
        selectedTrans = trans[row * B:(row + 1) * B, col * B:(col + 1) * B]
        N255 = Normalize(0, 255)
        plt.imshow(selectedTrans, interpolation='nearest')
        plt.colorbar(shrink=0.5)
        plt.title("DCT Image")
        plt.show()
        back = np.zeros((h, w), np.float32)
        for row in range(sV):
            for col in range(sH):
                currentblock = cv2.idct(trans[row * B:(row + 1) * B, col * B:(col + 1) * B])
                back[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
        cv2.cv.SaveImage('BackTransfromed.jpg', cv2.cv.fromarray(back))

        diff = back - im1
        print(diff.max())
        print(diff.min())
        MAD = np.sum(np.abs(diff)) / float(h * w)
        print("Mean Absolute Difference: ", MAD) 
        plt.imshow(back, cmap="gray")
        plt.figure()
        plt.title("Backtransformed Image")
        plt.show()"""
def Radon():
            im = cv2.imread(name)
            #img = im - np.mean(im)
            rad = radon(im, theta=[0, 45, 90])
            cv2.imshow("Radonn", rad)
def FFT():
        global FFT
        img = cv2.imread(name,0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        FFT = 20 * np.log(np.abs(fshift))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(FFT, cmap='gray')
        plt.title('FFT'), plt.xticks([]), plt.yticks([])
        plt.show()

def Gabor():
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0,
        np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor = cv2.filter2D(img, cv2.CV_8UC3, gabor_kernel)
    cv2.imshow("Orjinal Resim", img)
    cv2.imshow("Gabor", gabor)
    h, w = gabor_kernel.shape[:2]
    g_kernel = cv2.resize(gabor_kernel, (3 * w, 3 * h),
                    interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)

    """def gabor(sigma, theta, Lambda, psi, gamma):
            sigma_x = sigma
            sigma_y = float(sigma) / gamma

            # Bounding box
            nstds = 3  # Number of standard deviation sigma
            xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
            xmax = np.ceil(max(1, xmax))
            ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
            ymax = np.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

            # Rotation
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)

            gb = np.exp(im,-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
                2 * np.pi / Lambda * x_theta + psi)
            return gb"""
    #if combo1.get() == "Dalgacik":
def Dalgacik():

        im = cv2.imread(name,0)
        # Wavelet transform of image, and plot approximation and details
        titles = ['Approximation', ' Horizontal ','Vertical ', 'Diagonal ']
        wavelet = pywt.dwt2(im, 'bior1.3')
        LL, (LH, HL, HH) = wavelet
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap="gray")
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()

def LBP():
        global features
        img = cv2.imread(name)
        (h, w) = img.shape[:2]
        width = 160
        ratio = width / float(w)
        dim = (width, int(h * ratio))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cellSize = h / 10
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        cv2.imshow("Gray", gray) 
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots()
        fig.suptitle("Local Binary Patterns")
        plt.ylabel("% of Pixels")
        plt.xlabel("LBP pixel bucket")
        features = feature.local_binary_pattern(gray, 10, 5, method="default")
        cv2.imshow("LBP", features.astype("uint8"))
        ax.hist(features.ravel(), normed=True, bins=20, range=(0, 256))
        ax.set_xlim([0, 256])
        ax.set_ylim([0, 0.030])
        plt.show()

        """stacked = np.dstack([gray] * 3)

        # Divide the image into 100 pieces
        (h, w) = stacked.shape[:2]
        cellSizeYdir = h / 10
        cellSizeXdir = w / 10

        # Draw the box around area
        # loop over the x-axis of the image
        for x in xrange(0, w, cellSizeXdir):
            # draw a line from the current x-coordinate to the bottom of
            # the image

            cv2.line(stacked, (x, 0), (x, h), (0, 255, 0), 1)
            #   
        # loop over the y-axis of the image
        for y in xrange(0, h, cellSizeYdir):
            # draw a line from the current y-coordinate to the right of
            # the image
            cv2.line(stacked, (0, y), (w, y), (0, 255, 0), 1)

        # draw a line at the bottom and far-right of the image
        cv2.line(stacked, (0, h - 1), (w, h - 1), (0, 255, 0), 1)
        cv2.line(stacked, (w - 1, 0), (w - 1, h - 1), (0, 255, 0), 1)"""


"""def cagir():
    while True:
        a = Thread(target=hibritle, args=("a", 1))
        b = Thread(target=hibritle, args=("b", 0.5))
        c = Thread(target=hibritle, args=("c", 3))
        a.start()
        b.start()
        c.start()"""
"""def hibritle():
    if (checkbox==True ,checkbox1==True):

        plt.imshow("a")


    if(checkbox==True, checkbox2==True):
        im = cv2.imread(name, 0)

checkbox = tk.Checkbutton(pencere, text="FFT",bg="thistle")
checkbox.place(relx=0.5, rely=0.7, anchor="s")
checkbox1 = tk.Checkbutton(pencere, text="LBP",bg="thistle")
checkbox1.place(relx=0.6, rely=0.7, anchor="s")
checkbox2 = tk.Checkbutton(pencere, text="Gabor",bg="thistle")
checkbox2.place(relx=0.5, rely=0.8, anchor="s")
checkbox3 = tk.Checkbutton(pencere, text="DCT",bg="thistle")
checkbox3.place(relx=0.6, rely=0.8, anchor="s")
checkbox4 = tk.Checkbutton(pencere, text="Dalgacık",bg="thistle")
checkbox4.place(relx=0.5, rely=0.9, anchor="s")

combobox = Combobox(pencere, width=8, values=["90", "180", "270"])
combobox.place(relx=0.2, rely=0.1, anchor="s")
combo = Combobox(pencere, width=8,values=["Median", "Mean", "Gaussian", "Prewit", "Sobel", "Laplacian", "Min", "Max"])
combo.place(relx=0.4, rely=0.4, anchor="s")
combo1 = Combobox(pencere, width=8, values=["FFT", "DCT", "Radon", "Dalgacik", "LBP", "Gabor"])
combo1.place(relx=0.5, rely=0.4, anchor="s")"""
"""# Butonlar
button1 = tk.Button(pencere, text="Dosya Seç", width=10, height=3, command=resimac, justify="center")
button1.place(relx=0.1, rely=0.2, anchor="s")
button2 = tk.Button(pencere, text="Resmi Döndür", width=10, height=3, command=dondur, justify="center")
button2.place(relx=0.2, rely=0.2, anchor="s")

text1 = tk.Label(pencere, text="Histogram",bg="thistle")
text1.place(relx=0.1, rely=0.3, anchor="s")
button3 = tk.Button(pencere, text="Griye Çevir", width=10, height=3, command=gri)
button3.place(relx=0.1, rely=0.4, anchor="s")
button4 = tk.Button(pencere, text="Sabit Değer", width=10, height=3, command=histogram, justify="center")
button4.place(relx=0.1, rely=0.5, anchor="s")
button12 = tk.Button(pencere, text="Eşikleme", width=10, height=3, command=esikleme, justify="center")
button12.place(relx=0.1, rely=0.6, anchor="s")
button13 = tk.Button(pencere, text="Otsu", width=10, height=3, command=otsu, justify="center")
button13.place(relx=0.1, rely=0.7, anchor="s")

text2 = tk.Label(pencere, text="Morfolojik İşlemler",bg="thistle")
text2.place(relx=0.2, rely=0.3, anchor="s")
button5 = tk.Button(pencere, text="Erozyon", width=10, height=3, command=asinma, justify="center")
button5.place(relx=0.2, rely=0.4, anchor="s")
button6 = tk.Button(pencere, text="Dilation", width=10, height=3, command=genisleme, justify="center")
button6.place(relx=0.2, rely=0.5, anchor="s")
button7 = tk.Button(pencere, text="Açma", width=10, height=3, command=acma, justify="center")
button7.place(relx=0.2, rely=0.6, anchor="s")
button8 = tk.Button(pencere, text="Kapama", width=10, height=3, command=kapama, justify="center")
button8.place(relx=0.2, rely=0.7, anchor="s")

text3 = tk.Label(pencere, text="Hough Dönüşümü",bg="thistle")
text3.place(relx=0.3, rely=0.3, anchor="s")
button9 = tk.Button(pencere, text="Gradyan", width=10, height=3, command=gradyan, justify="center")
button9.place(relx=0.3, rely=0.4, anchor="s")
button10 = tk.Button(pencere, text="Daire", width=10, height=3, command=daire, justify="center")
button10.place(relx=0.3, rely=0.5, anchor="s")
button11 = tk.Button(pencere, text="Çizgi", width=10, height=3, command=cizgi, justify="center")
button11.place(relx=0.3, rely=0.6, anchor="s")

# Filtreler
text4 = tk.Label(pencere, text="Filtreler",bg="thistle")
text4.place(relx=0.4, rely=0.3, anchor="s")
button16 = tk.Button(pencere, text="Uygula", width=10, height=3, command=filtreler, justify="center")
button16.place(relx=0.4, rely=0.5, anchor="s")
button17 = tk.Button(pencere, text="Kümeleme", width=10, height=3, command=kumeleme, justify="center")
button17.place(relx=0.4, rely=0.6, anchor="s")
button18 = tk.Button(pencere, text="Karşıtlık", width=10, height=3, command=karsitlik, justify="center")
button18.place(relx=0.4, rely=0.7, anchor="s")
text5 = tk.Label(pencere, text="Öznitelikler",bg="thistle")
text5.place(relx=0.5, rely=0.3, anchor="s")
button19 = tk.Button(pencere, text="Uygula", width=10, height=3, command=oznitelikler, justify="center")
button19.place(relx=0.5, rely=0.5, anchor="s")
button20 = tk.Button(pencere, text="Hibrit", width=10, height=3, command=hibritle, justify="center")
button20.place(relx=0.5, rely=0.6, anchor="s")"""

menu = tk.Menu(pencere)

anaMenu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Dosya", menu=anaMenu)
anaMenu.add_command(label="Resim Aç", command=resimac)
anaMenu.add_command(label="Döndür", command=dondur)
anaMenu.add_command(label="Griye Çevir", command=gri)
#anaMenu.add_separator()

hist=tk.Menu(menu,tearoff=0)
menu.add_cascade(label="Histogram", menu=hist)
hist.add_command(label="Sabit Değer", command=histogram)
hist.add_command(label="Eşikleme", command=esikleme)
hist.add_command(label="Otsu", command=otsu)
#hist.add_separator()

morf=tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Morfolojik İşlemler", menu=morf)
morf.add_command(label="Erosion", command=asinma)
morf.add_command(label="Dilation", command=genisleme)
morf.add_command(label="Açma", command=acma)
morf.add_command(label="Kapama", command=kapama)
morf.add_command(label="Gradyan", command=gradyan)
#morf.add_separator()

hough=tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Hough", menu=hough)
hough.add_command(label="Daire", command=daire)
hough.add_command(label="Çizgi", command=cizgi)

filtre=tk.Menu(menu,tearoff=0)
menu.add_cascade(label="Filtreler", menu=filtre)
filtre.add_command(label="Median", command=median)
filtre.add_command(label="Mean", command=mean)
filtre.add_command(label="Gauissian", command=gaussian)
filtre.add_command(label="Prewitt", command=prewitt)
filtre.add_command(label="Sobel", command=sobel)
filtre.add_command(label="Laplacian", command=laplacian)
filtre.add_command(label="Min", command=min)
filtre.add_command(label="Max", command=max)

kume=tk.Menu(menu,tearoff=0)
menu.add_cascade(label="Kümeleme",menu=kume)
kume.add_command(label="Kümele", command=kumeleme)

karsit=tk.Menu(menu,tearoff=0)
menu.add_cascade(label="Karşıtlık", menu=karsit)
karsit.add_command(label="Karşıtlık Yayma", command=karsitlik)

oznitelik=tk.Menu(menu,tearoff=0)
menu.add_cascade(label="Öznitelikler",menu=oznitelik)
oznitelik.add_command(label="FFT", command=FFT)
oznitelik.add_command(label="DCT", command=DCT)
oznitelik.add_command(label="Gabor", command=Gabor)
oznitelik.add_command(label="Radon", command=Radon)
oznitelik.add_command(label="LBP", command=LBP)

pencere.config(menu=menu)
pencere.mainloop()
cv2.waitKey(0)
cv2.destroyAllWindows()
