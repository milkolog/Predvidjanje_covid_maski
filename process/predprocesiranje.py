import cv2
import os


def smanjivanje_dimenzija_slike(img_path, dimenzija_x=224, dimenzija_y=224):
    # da bismo trenirali model najbolje bi bilo da slike budu manje nego 1024*1024
    img = cv2.imread(img_path)
    img = cv2.resize(img, (dimenzija_x, dimenzija_y))
    return img


if __name__ == "__main__":
    # ucitavanje podataka
    path = r"..\input"
    path_pravilno = path + r"\pravilno_nosene"
    path_nepravilno = path + r"\nepravilno_nosene"

    img = cv2.imread(path_pravilno + r"\38000_Mask.jpg")
    cv2.imshow(".", img)
    cv2.waitKey(0)
    # cv2.imwrite(fr"..\input\skalirane\pravilno\mask_0.jpg", img)

    cnt = 0
    # for za pravilne
    # for i in range(38000, 39000):
    #
    #     img_path = path_pravilno + rf"\{i}_Mask.jpg"
    #
    #     if os.path.exists(img_path):
    #         # img = cv2.imread(img_path)
    #         img = smanjivanje_dimenzija_slike(img_path)
    #         # cv2.imshow(".", img)
    #         # cv2.waitKey(0)
    #
    #         cv2.imwrite(fr"..\input\skalirane\pravilno\mask_{cnt}.jpg", img)
    #         cnt += 1

    # brojaci su tu da bismo imali kontinuitet u folderu sa slikama
    n1 = 0
    n2 = 191
    n3 = 291
    # for za nepravilne
    for i in range(64000, 65000):

        img_path_nos = path_nepravilno + rf"\{i}_Mask_Mouth_Chin.jpg"
        img_path_nos_usta = path_nepravilno + rf"\{i}_Mask_Chin.jpg"
        img_path_brada = path_nepravilno + rf"\{i}_Mask_Nose_Mouth.jpg"

        if os.path.exists(img_path_nos):
            pass
            # img = cv2.imread(img_path)
            img = smanjivanje_dimenzija_slike(img_path_nos)
            # cv2.imshow(".", img)
            # cv2.waitKey(0)

            cv2.imwrite(fr"..\input\skalirane\nepravilno_nos\mask_nos_{n1}.jpg", img)
            n1 += 1
        elif os.path.exists(img_path_nos_usta):
            img = smanjivanje_dimenzija_slike(img_path_nos_usta)
            cv2.imwrite(fr"..\input\skalirane\nepravilno_nos_usta\mask_nos_usta_{n2}.jpg", img)
            n2 += 1
        elif os.path.exists(img_path_brada):
            img = smanjivanje_dimenzija_slike(img_path_brada)
            cv2.imwrite(fr"..\input\skalirane\nepravilno_brada\mask_brada_{n3}.jpg", img)
            n3 += 1
        else:
            pass
    # zadrzavamo imena samo im menjamo folder u skalirano i sve slike ce biti na jednom mestu

    # u sledecoj skripti ce se iz slika izvlaciti nazivi i dodavati labele u df
