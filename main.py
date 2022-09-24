import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


# 2) Выделить из полноцветного изображения каждый из каналов R, G, B  и вывести результат.
# Построить гистограмму по цветам (3 штуки).
def GetCurrentChanel(image_path, color):
    try:
        img = cv2.imread(image_path)
        if color == "blue":
            # blue_channel = img[:, :, 0]
            # blue_img = np.zeros(img.shape)
            # blue_img[:, :, 0] = blue_channel
            # cv2.imwrite(os.getcwd() + "\dog_blue.jpg", blue_img)
            # cv2.imshow("Dog_blue", cv2.imread(os.getcwd() + "\dog_blue.jpg"))
            # cv2.waitKey()
            for y in range(0, img.shape[0]):
                for x in range(0, img.shape[1]):
                    (b, g, r) = img[y, x]
                    img[y, x] = (b, 0, 0)
            cv2.imwrite(os.getcwd() + "\dog_blue.jpg", img)
            cv2.imshow("Dog_blue", cv2.imread(os.getcwd() + "\dog_blue.jpg"))
            cv2.waitKey()
        elif color == "green":
            for y in range(0, img.shape[0]):
                for x in range(0, img.shape[1]):
                    (b, g, r) = img[y, x]
                    img[y, x] = (0, g, 0)
            cv2.imwrite(os.getcwd() + "\dog_green.jpg", img)
            cv2.imshow("Dog_green", cv2.imread(os.getcwd() + "\dog_blue.jpg"))
            cv2.waitKey()
            # green_channel = img[:, :, 1]
            # green_img = np.zeros(img.shape)
            # green_img[:, :, 1] = green_channel
            # cv2.imwrite(os.getcwd() + "\dog_green.jpg", green_img)
            # cv2.imshow("Dog_green", cv2.imread(os.getcwd() + "\dog_green.jpg"))
            # cv2.waitKey()
        elif color == "red":
            for y in range(0, img.shape[0]):
                for x in range(0, img.shape[1]):
                    (b, g, r) = img[y, x]
                    img[y, x] = (0, 0, r)
            cv2.imwrite(os.getcwd() + "\dog_red.jpg", img)
            cv2.imshow("Dog_red", cv2.imread(os.getcwd() + "\dog_red.jpg"))
            cv2.waitKey()
            # red_channel = img[:, :, 2]
            # red_img = np.zeros(img.shape)
            # red_img[:, :, 2] = red_channel
            # cv2.imwrite(os.getcwd() + "\dog_red.jpg", red_img)
            # cv2.imshow("Dog_red", cv2.imread(os.getcwd() + "\dog_red.jpg"))
            # cv2.waitKey()
    except TypeError as er:
        print(f"File not found!")
        return
    print("Files created! " + "dog_" + f"{color}.jpg create as {os.getcwd()}")


def ShowHistograms(img_path, type_color):
    image = cv2.imread("dog_" + f"{type_color}.jpg")
    plt.figure(figsize=(10,10)) # size of figure
    plt.xlim([0, 256]) # intensity of each color in range [0,255]

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    # image[:,:,i] where i = [0,1,2] means channel(0 is blue, 1 is green, 2 is red)

    if type_color == "blue":
        histogram, bin_edges = np.histogram(image[:, :, 0], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color="blue")
        plt.show()
        exit()
    elif type_color == "green":
        histogram, bin_edges = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color="green")
        plt.show()
        exit()
    elif type_color == "red":
        histogram, bin_edges = np.histogram(image[:, :, 2], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color="red")
        plt.show()
        exit()


# in cv2 we have BGR format,not RGB
if __name__ == '__main__':
    print("Lab2, task2")
    list_color = ["blue", "green", "red"]
    image_path = "Image/dog.jpg"

    print("Input the color: ")
    _color = input()
    if _color not in list_color:
        raise Exception("Incorrect color!") # check _color is list colors

    GetCurrentChanel(image_path=image_path, color=_color) # get current channel and created img in this channel
    ShowHistograms(image_path, type_color=_color) # show histograms by selected color
