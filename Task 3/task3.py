import os
from  math import floor
import tkinter as tk
from math import floor
from tkinter import filedialog as fdiag
from tkinter import TOP,CENTER, messagebox
from numba import njit, prange
import numpy as np
from PIL import Image
from PIL import ImageTk as imagetk


class HueSaturationValue(tk.Frame):
    img_data: np.ndarray
    hsv_data: np.array
    orig: np.ndarray
    H = 0
    S = 0
    V = 0
    WIN_H = 500
    WIN_W = 700
    PANEL_W = 200
    PANEL_H = WIN_H
    CANVAS_H = WIN_H
    CANVAS_W = WIN_W - PANEL_W

    def __init__(self, parent): # fixed : AttributeError: '_tkinter.tkapp' object has no attribute 'PassCheck'
        tk.Frame.__init__(self,parent)
        self.parent = parent
        self.parent.title("RGB to HSV")
        self.parent.config(width=self.WIN_W, height=self.WIN_H)
        self.parent.resizable(False, False)
        self.AddedWidgets()

    def AddedWidgets(self):
        self.panel = tk.Frame(self.parent, width=self.PANEL_W, height=self.PANEL_H, background='white')
        self.panel.place(x=0, y=0, width=self.PANEL_W, height=self.PANEL_H)
        self.canvas = tk.Canvas(self.parent, width=self.CANVAS_W, height=self.CANVAS_H, background='white')
        self.canvas.create_text(self.CANVAS_W/2,self.CANVAS_H/2,text="NO DATA",justify=CENTER,font="Verdana 25")

        self.save_file = tk.Button(self.panel, text="Save origin image", command=self.SaveTheFile)
        self.file_open = tk.Button(self.panel, text="Load the sea image", command=self.LoadTheFile)

        self.slider_hue = tk.Scale(self.panel, from_=0, to_=359, orient=tk.HORIZONTAL, command=self.ChangedHue,
                                   label="Hue")
        self.slider_saturation = tk.Scale(self.panel, from_=0, to_=200, orient=tk.HORIZONTAL,
                                          command=self.ChangedSaturation,
                                          label="Saturation")
        self.slider_value = tk.Scale(self.panel, from_=0, to_=200, orient=tk.HORIZONTAL, command=self.ChangedValue,
                                     label="Value")

        self.file_open.pack(padx=5, pady=5)
        self.save_file.pack(padx=5, pady=5)

        self.slider_hue.pack(side=TOP)
        self.slider_saturation.pack(side=TOP)
        self.slider_value.pack(side=TOP)

        self.canvas.place(x=self.PANEL_W, y=0, width=self.CANVAS_W, height=self.CANVAS_H)

        self.slider_value.set(100) # убрать затенение

    def LoadTheFile(self):
        path_to_image = os.getcwd() + '\sea.jpg'
        img = Image.open(path_to_image)
        self.orig = np.asarray(img).copy()
        self.img_data = np.asarray(img).copy()
        self.hsv_data = ConvertRGBToHSV(self.img_data, np.empty_like(self.img_data).astype(np.float64))
        self.UpdateTheImage()

    def SaveTheFile(self):
        Image.fromarray(self.orig).save(fdiag.asksaveasfile())

    def UpdateTheImage(self):
        self.image = imagetk.PhotoImage(Image.fromarray(self.img_data))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def ChangedHue(self, hue):
        self.H = int(hue)
        self.img_data = ConvertHVStoRGB(self.hsv_data, self.img_data, self.H, self.S, self.V)
        self.UpdateTheImage()

    def ChangedSaturation(self, saturation):
        self.S = int(saturation)
        self.img_data = ConvertHVStoRGB(self.hsv_data, self.img_data, self.H, self.S, self.V)
        self.UpdateTheImage()

    def ChangedValue(self, value):
        self.V = int(value)
        self.img_data = ConvertHVStoRGB(self.hsv_data, self.img_data, self.H, self.S, self.V)
        self.UpdateTheImage()


@njit(parallel=True, cache=True)
def ConvertRGBToHSV(ImgData: np.array, NewImgData: np.array) -> np.array:
    for j in prange(0, ImgData.shape[0]):
        for i in prange(0, ImgData.shape[1]):
            NewImgData[j][i] = ChangeHSVpixel(ImgData[j][i])
    return NewImgData


@njit(cache=True)
def get_rgb_pixel(pixel: np.array) -> np.array:
    H, S, V = pixel
    Hi = floor(H / 60) % 6
    f = H / 60 - floor(H / 60)
    p = V * (1. - S)
    q = V * (1. - f * S)
    t = V * (1. - (1. - f) * S)

    match Hi:
        case 0:
            return np.array([V, t, p])
        case 1:
            return np.array([q, V, p])
        case 2:
            return np.array([p, V, t])
        case 3:
            return np.array([p, q, V])
        case 4:
            return np.array([t, p, V])
        case 5:
            return np.array([V, p, q])

    return np.empty(3)


@njit(parallel=True, cache=True)
def ConvertHVStoRGB(first, last, h, s, v):
    for j in prange(0, first.shape[0]):
        for i in prange(0, first.shape[1]):
            H = (first[j][i][0] + h) % 360.0
            S = save_elem(first[j][i][1] * (s / 100.), 0.0, 1.0)
            V = save_elem(first[j][i][2] * (v / 100.), 0.0, 1.0)
            last[j][i] = get_rgb_pixel(np.array([H, S, V])) * 255
    return last



@njit(cache=True, inline='always')
def save_elem(elem, min_value, max_value) -> float:
    if (elem > max_value):
        return max_value
    if (elem < min_value):
        return min_value
    return elem


@njit(cache=True)
def ChangeHSVpixel(pixel):
    r, g, b = pixel / 255
    max_value = max(r, g, b)
    min_value = min(r, g, b)
    H, S, V = [0.0, 0.0, 0.0]
    if (max_value == min_value):
        H = 0.0
    elif (max_value == r and g >= b):
        H = 60.0 * (g - b) / (max_value - min_value)
    elif (max_value == r and g < b):
        H = 60.0 * (b - r) / (max_value - min_value) + 360.0
    elif (max_value == g):
        H = 60.0 * (b - r) / (max_value - min_value) + 120.0
    elif (max_value == b):
        H = 60.0 * (r - g) / (max_value - min_value) + 240.0

    if max_value == 0:
        S = 0.0
    else:
        S = 1.0 - min_value / max_value

    V = max_value

    return np.array([H, S, V])


if __name__ == '__main__':
    print("Laba2, task3")
    root = tk.Tk()
    HueSaturationValue(root)
    root.mainloop()