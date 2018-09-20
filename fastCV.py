import cv2
import numpy as np
import copy

B = 0
R = 1
G = 2
A = 1

def newImg(size, depth, channel):
    if depth == 16:
        adepth = np.uint16
    else:
        if depth == 8:
            adepth = np.uint8
        else:
            if depth == 32:
                adepth = np.uint32
            else:
                raise AssertionError("depth not exist")
    if channel == "RGBA":
        achannel = 4
    else:
        if channel == "RGB":
            achannel = 3
        else:
            if channel == "Gray":
                achannel = 1
            else:
                raise AssertionError("channel not exist")
    pic = (size[0], size[1], achannel)
    img = np.zeros(pic, adepth)
    return img


def newCvImg(size, depth, channel):
    return cvImg(newImg(size, depth, channel))


class cvWindows:
    def __init__(self, name):
        self.name = name
        cv2.namedWindow(name)

    def show(self, img):
        cv2.imshow(self.name, img)


class cvImg:
    def __init__(self, obj):
        if isinstance(obj, str):
            self.img = cv2.imread(obj)
        else:
            self.img = copy.copy(obj)
        self.channels = self.img.Channels  # 这个和带n的都测试一下看看有没有

    def save(self, path):
        cv2.imwrite(path, self.img)

    def getCapture(self):
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        if success:
            return frame
        else:
            return False

    def rect(self, start, end, color):
        cv2.rectangle(self.img, start, end, color)

    def circle(self, pos, r, color):
        cv2.circle(self.img, pos, r, color)

    def line(self, start, end, color, width=1):
        cv2.line(self.img, start, end, color, width)

    def toGray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.channels = 1

    def toBinarization(self, threshold, bigBlack = False):
        if self.channels != 1:
            self.toGray()
        if bigBlack:
            ret, self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)
        self.channels = 0

    def foreachColor(self, func):
        if self.channels == 1:
            for i in range(self.img.height):
                for j in range(self.img.width):
                    self.img[i][j] = func((i, j), self.img[i][j])
            return
        if self.channels == 3:
            for i in range(self.img.height):
                for j in range(self.img.width):
                    r = self.img[i][j][R]
                    g = self.img[i][j][G]
                    b = self.img[i][j][B]
                    (self.img[i][j][R], self.img[i][j][G], self.img[i][j][B]) = func((i, j), (r, g, b))
            return
        if self.channels == 4:
            for i in range(self.img.height):
                for j in range(self.img.width):
                    r = self.img[i][j][R]
                    g = self.img[i][j][G]
                    b = self.img[i][j][B]
                    a = self.img[i][j][A]
                    (self.img[i][j][R], self.img[i][j][G], self.img[i][j][B], self.img[i][j][A]) = func((i, j),
                                                                                                        (r, g, b, a))
            return

    def setColor(self, pos, color):  # None的元素指保持原样，必须有一个元素非None
        if len(color) == 1:
            if self.channels != 1:
                raise AssertionError("color len not same as channels")
            self.img[pos[0]][pos[1]] = color[1]
            return
        if not (color[0] is None):
            self.img[pos[0]][pos[1]][R] = color[0]
        if not (color[1] is None):
            self.img[pos[0]][pos[1]][G] = color[1]
        if not (color[2] is None):
            self.img[pos[0]][pos[1]][B] = color[2]
        if len(color) == 4:
            if not (color[3] is None):
                if self.channels != 4:
                    raise AssertionError("color len not same as channels")
            self.img[pos[0]][pos[1]][A] = color[3]

    def getColor(self, pos, color):  # color指颜色类型
        return self.img[pos[0]][pos[1]][color]

    def blur(self, Range):
        cv2.boxFilter(self.img, -1, Range)  # Range是二元组，十字大小

    def sobelSketch(self):
        x = cv2.Sobel(self.img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(self.img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        self.img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def edgeDetection(self, max, min):
        self.img = cv2.GaussianBlur(self.img, (3, 3), 0)
        if self.channels != 1:
            self.toGray()
        self.img = cv2.Canny(self.img, min, max)

    def lineDetection(self, max, min, accurate):  # 直接在原图上画线，不改变颜色通道
        edge = copy.copy(self)
        edge.edgeDetection(max, min)

        if accurate:
            lines = cv2.HoughLinesP(edge.img, 1, np.pi / 180, 80, 200, 15)
            for x1, y1, x2, y2 in lines:
                cv2.line(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return

        lines = cv2.HoughLines(edge.img, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
        for line in lines:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - self.img.shape[0] * np.sin(theta)) / np.cos(theta)), self.img.shape[0])
                # 绘制一条白线
                cv2.line(self.img, pt1, pt2, (255))
            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (self.img.shape[1], int((rho - self.img.shape[1] * np.cos(theta)) / np.sin(theta)))
                # 绘制一条直线
                cv2.line(self.img, pt1, pt2, (255), 1)

    def equalization(self):
        self.img = cv2.equalizeHist(self.img)

    def contourDetection(self, threshold, color):
        self.toBinarization(threshold)
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.img, contours, -1, color, 3)

    def colorDelete(self, reservedColor, replacedColor):
        def func(pos, color):
            if color != reservedColor:
                return replacedColor
            return color

        self.foreachColor(func)