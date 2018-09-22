import cv2
import help
import numpy as np
import copy

B = 0
R = 1
G = 2
A = 1


class cvWindows:
    def __init__(self, name):
        self.name = name
        cv2.namedWindow(name)

    def show(self, img):
        cv2.imshow(self.name, img)

def read(path):
    return cv2.imread(path)

def save(img,path):
    cv2.imwrite(path,img)

def getCapture():  # 从摄像头获取数据
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if success:
        return frame
    else:
        return False

def drawRect(img, start, end, color):
    cv2.rectangle(img, start, end, color)

def drawCircle(img, pos, r, color, thick=None): # 最后一个参数为线的厚度，负值为填充
    cv2.circle(img, pos, r, color, thickness = thick)

def drawPoint(img, pos, color, radius=2):
    drawCircle(img, pos, radius, color, thick = -1)

def drawLine(img, start, end, color, width=1):
    cv2.line(img, start, end, color, width)

def toGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # channels=1

def getChannels(img):
    return img.channels

def toBinarization(img, threshold, maxthreshold=255, bigBlack=False):
    if getChannels(img) != 1:
        toGray(img)
    if bigBlack:
        ret, img = cv2.threshold(img, threshold, maxthreshold, cv2.THRESH_BINARY_INV)  # 这里把最大值固定了，某些特殊情况需要过滤过亮的像素就要改
    else:
        ret, img = cv2.threshold(img, threshold, maxthreshold, cv2.THRESH_BINARY)
    # channels=0
    return img


def foreachColor(img, func):  # 对每个像素执行func，func返回改像素修改后的值
    if getChannels(img) == 1:
        for i in range(img.height):
            for j in range(img.width):
                img[i][j] = func((i, j), img[i][j])
        return
    if getChannels(img) == 3:
        for i in range(img.height):
            for j in range(img.width):
                r = img[i][j][R]
                g = img[i][j][G]
                b = img[i][j][B]
                (img[i][j][R], img[i][j][G], img[i][j][B]) = func((i, j), (r, g, b))
        return
    if getChannels(img) == 4:
        for i in range(img.height):
            for j in range(img.width):
                r = img[i][j][R]
                g = img[i][j][G]
                b = img[i][j][B]
                a = img[i][j][A]
                (img[i][j][R], img[i][j][G], img[i][j][B], img[i][j][A]) = func((i, j), (r, g, b, a))
        return


def setColor(img, pos, color):
    if len(color) == 1:
        if getChannels(img) != 1:
            raise AssertionError("color len not same as channels")
        img[pos[0]][pos[1]] = color[1]
        return

    # None表示该像素值不变，不是None才修改
    if not (color[0] is None):
        img[pos[0]][pos[1]][R] = color[0]
    if not (color[1] is None):
        img[pos[0]][pos[1]][G] = color[1]
    if not (color[2] is None):
        img[pos[0]][pos[1]][B] = color[2]
    if len(color) == 4:
        if not (color[3] is None):
            if getChannels(img) != 4:
                raise AssertionError("color len not same as channels")
        img[pos[0]][pos[1]][A] = color[3]


def getColor(img, pos, color):  # color指颜色通道
    return img[pos[0]][pos[1]][color]

def boxBlur(img, size):  # 低通滤波，滤波器中每个像素的权重是相同的。
    return cv2.boxFilter(img, -1, size)  # size是二元组，方块滤波（模糊）窗口大小

def gaussianBlur(img, size):  # 高斯滤波，像素的权重与其距中心像素的距离成比例
    return cv2.GaussianBlur(img, size, 0)

def medianBlur(img, sizeRadius):  # 中值滤波，椒盐现象不会影响滤波结果，如果在某个像素周围有白色或黑色的像素，这些白色或黑色的像素不会
                                # 选择作为中值（最大或最小值不用），而是被替换为邻域值。滤波窗口大小（孔径尺寸）为sizeRadius*sizeRadius
    return cv2.medianBlur(img, sizeRadius)

def fusion(img1, img2, ratio1, ratio2):  # 图像融合
    return cv2.addWeighted(img1, ratio1, img2, ratio2, 0.0)

# 以下的几个边缘检测算法都可以先做个高斯模糊去噪点
def sobelSketch(img):  # sobel算子边缘检测（是一种更接近素描化的效果）
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，
                                            # 所以sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转换回int8类型，否则将无法显示图像，而只是一副灰色的窗口
    absY = cv2.convertScaleAbs(y)
    return fusion(absX, 0.5, absY, 0.5)  # 先前sobel算子在两个方向计算，这里进行图像融合


def laplacianSketch(img):  # laplacian边缘检测
    gray_lap = cv2.Laplacian(img, cv2.CV_16S)  # ksize这个参数暂不设为可调
    return cv2.convertScaleAbs(gray_lap)

def cannySketch(img, max, min):  # canny算子边缘检测（都是细线，效果最佳）
    if getChannels(img) != 1:  # canny只能处理灰度图，所以转成灰度
        img = toGray(img)
    return cv2.Canny(img, min, max)  # 小阈值用来控制边缘连接，大的阈值用来控制强边缘的初始分割。即如果一个像素的
                                    # 梯度大于上限值，则被认为是边缘像素，如果小于下限阈值，则被抛弃。如果该点
                                    # 的梯度在两者之间则当这个点与高于上限值的像素点连接时我们才保留，否则删除。

def globalEqualization(img): # 全局直方图均衡化
    return cv2.equalizeHist(img)

def partialEqualization(img, size=None, threshold=None): # 局部（分成多个小块分别）进行均衡化，threshold为对比度限制阈值
    clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=size)
    return clahe.apply(img)

def contourDetection(img, threshold, color = (0,0,255)):
    img = toBinarization(img, threshold)
    img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(img, contours, -1, color, 3)  # 第三个参数可以控制绘制哪条轮廓。每个轮廓的数据为contours中的元素，每
                                                          # 个元素里是该轮廓的顶点

def getChannelHistogram(img):
    if getChannels(img) < 3:
        raise AssertionError("color channel must be greater than or equal to 3.")

    b, g, r = cv2.split(img)
    hb = help.calcAndDrawHist(b, [255, 0, 0])
    hg = help.calcAndDrawHist(g, [0, 255, 0])
    hr = help.calcAndDrawHist(r, [0, 0, 255])
    return hr, hg, hb


def getHistogram(img):
    h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像
    bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
    for ch, col in enumerate(color):
        originHist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
        hist = np.int32(np.around(originHist))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    return np.flipud(h)

# 建议先做高斯模糊去噪点
def lineDetection(img, edgemax=150, edgemin=50, minLineLength=60, maxLineGap=10, votesThreshold=50, lineColor=(0, 255, 0)):  # 霍夫直线检测，直接在原图上画线，
                                                                                                                    # maxLineGap为同一线路上的点之间的最大允许间隙
                                                                                                                    # votesThreshold为选出直线的最低票数
    edge = copy.copy(img)
    edge = cannySketch(edge, edgemax, edgemin)
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, votesThreshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    for x1, y1, x2, y2 in lines[:, 0, :]:  # 提取为二维再搞
        drawLine(img, (x1, y1), (x2, y2), lineColor)


def circlesDetection(img, minDist=70, minRadius=None, maxRadius=None, lineColor=(255, 0, 0)):  # 霍夫圆检测，直接在原图上画圆，minDist为最小圆间距
    gray = copy.copy(img)
    gray = toGray(gray)
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, minRadius=minRadius, maxRadius=maxRadius)
    circles = circles1[0, :, :]  # 提取为二维
    circles = np.uint16(np.around(circles))  # 四舍五入，取整
    for i in circles:
        drawCircle(img, (i[0], i[1]), i[2], lineColor)  # 画圆
        drawPoint(img, (i[0], i[1]), lineColor)  # 画圆心