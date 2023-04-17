import cv2
import matplotlib.pyplot as plt
import numpy as np


# skripta za rotaciju i jos neke transformacije nad slikama da povecamo broj podataka slika
# mozemo da primenimo neke filtre kao sto su blur i izostravanje, rotacija, mirorovanje

def y_rotacija(img):
    # get the image shape
    rows, cols, dim = img.shape
    # transformation matrix for x-axis reflection
    # M = np.float32([[1,  0, 0   ],
    #                 [0, -1, rows],
    #                 [0,  0, 1   ]])
    # transformation matrix for y-axis reflection
    M = np.float32([[-1, 0, cols],
                    [0, 1, 0],
                    [0, 0, 1]])
    # apply a perspective transformation to the image
    reflected_img = cv2.warpPerspective(img, M, (int(cols), int(rows)))
    return reflected_img


def izostravanje(img):
    Sharpen_Kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    Sharpen_Effect_Img = cv2.filter2D(src=img, kernel=Sharpen_Kernel, ddepth=-1)
    return Sharpen_Effect_Img


def blurovanje(img):
    Blur_Effect_Img = cv2.GaussianBlur(img, (5, 5), 0)
    return Blur_Effect_Img


if __name__ == "__main__":
    cnt_nepravilno_brada = 385
    cnt_nepravilno_nos_usta = 261
    # img = cv2.imread(rf"..\input\skalirane\nepravilno_brada\mask_brada_0.jpg")
    # cv2.imshow("", img)
    for i in range(0, cnt_nepravilno_brada+1):
        img = cv2.imread(rf"..\input\skalirane\nepravilno_brada\mask_brada_{i}.jpg")
        # cv2.imshow("img.jpg", img)
        # cv2.waitKey(0)
        reflected_img = y_rotacija(img)

        # cv2.imshow("miror.jpg", reflected_img)
        # cv2.waitKey(0)
        cv2.imwrite(rf"..\input\skalirane\nepravilno_brada\mask_brada_{i+cnt_nepravilno_brada}.jpg", reflected_img)

    for i in range(0, cnt_nepravilno_nos_usta + 1):
        img = cv2.imread(rf"..\input\skalirane\nepravilno_nos_usta\mask_nos_usta_{i}.jpg")
        # cv2.imshow("img.jpg", img)
        # cv2.waitKey(0)
        reflected_img = y_rotacija(img)

        # cv2.imshow("miror.jpg", reflected_img)
        # cv2.waitKey(0)
        cv2.imwrite(rf"..\input\skalirane\nepravilno_nos_usta\mask_nos_usta_{i + cnt_nepravilno_nos_usta}.jpg", reflected_img)
