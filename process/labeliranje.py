import pandas as pd
import matplotlib.pyplot as plt


def labeilranje_foldera(folder_path, img_name, img_cnt, label=0):
    """Pravimo DataFrame na osnovu foldera koji sadrzi ime slike i kolonu koja oznacava kojoj labeli
    slika pripada
    0 - pravilno nose masku
    1 - maska ne pokriva nos
    2 - maska ne pokriva ni nos ni usta
    3 - maska ne pokriva bradu"""
    # todo: moze da se napravi da ne hardkocujemo brojac ali ukljucuje jos jednu petlju

    df = pd.DataFrame(columns=["putanja_slike", "labela"])

    for i in range(0, img_cnt + 1):
        df.loc[i, "putanja_slike"] = folder_path + rf"{img_name}_{i}.jpg"

    df['labela'] = label

    return df


if __name__ == "__main__":
    path = rf"..\input\undersampling"
    df_pravilno = labeilranje_foldera(path, r"\pravilno\mask", 197, 0)
    df_nepravilno_nos = labeilranje_foldera(path, r"\nepravilno_nos\mask_nos", 197, 1)
    df_nepravilno_nos_usta = labeilranje_foldera(path, r"\nepravilno_nos_usta\mask_nos_usta", 197, 2)
    df_nepravilno_brada = labeilranje_foldera(path, r"\nepravilno_brada\mask_brada", 197, 3)

    df_pravilno.to_csv(rf"..\input\labele\pravilno_under.csv", index=False)
    df_nepravilno_nos.to_csv(rf"..\input\labele\nepravilno_nos_under.csv", index=False)
    df_nepravilno_nos_usta.to_csv(rf"..\input\labele\nepravinlo_nos_usta_under.csv", index=False)
    df_nepravilno_brada.to_csv(rf"..\input\labele\nepravilno_brada_under.csv", index=False)

    df = pd.concat([df_pravilno, df_nepravilno_nos, df_nepravilno_nos_usta, df_nepravilno_brada],
                   ignore_index=True).reset_index().drop(['index'], axis=1)

    df.hist(['labela'])
    plt.show()
    print(df)
