import keras.saving.legacy.save
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout
from keras import applications
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def pravljenje_nizova_slika_i_labela(df):
    slike = []
    labele = []
    for i in range(len(df)):
        path = df.loc[i, "putanja_slike"]
        label = df.loc[i, "labela"]
        # kako da se oslobodim tipa podataka
        slika = cv2.imread(path)
        slike.append(slika)
        labele.append(label)
    slike = np.array(slike)
    labele = np.array(labele)
    # skaliranje, delimo sa 255 da bi matrica imala manje brojeve
    return slike/255, labele


def pretrenirani_model(dropout_rate=0.0, input_shape=(224, 224, 3)):
    base_model = applications.InceptionV3(weights='imagenet',
                                          include_top=False,
                                          input_shape=input_shape)
    base_model.trainable = False

    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(Flatten())
    # add_model.add(GlobalAveragePooling2D())
    add_model.add(Dense(128, activation='ReLU'))
    add_model.add(Dropout(dropout_rate))
    add_model.add(Dense(4, activation='softmax'))

    model = add_model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()

    print(model.summary())

    return model


if __name__ == "__main__":
    df_pravilno = pd.read_csv(rf"..\input\labele\pravilno.csv")
    df_nepravilno_nos = pd.read_csv(rf"..\input\labele\nepravilno_nos.csv")
    df_nepravilno_nos_usta = pd.read_csv(rf"..\input\labele\nepravinlo_nos_usta.csv")
    df_nepravilno_brada = pd.read_csv(rf"..\input\labele\nepravilno_brada.csv")

    slike_0, labele_0 = pravljenje_nizova_slika_i_labela(df_pravilno)
    slike_1, labele_1 = pravljenje_nizova_slika_i_labela(df_nepravilno_nos)
    slike_2, labele_2 = pravljenje_nizova_slika_i_labela(df_nepravilno_nos_usta)
    slike_3, labele_3 = pravljenje_nizova_slika_i_labela(df_nepravilno_brada)
    # print(slike_0)
    # print(labele_0)

    # 0
    X_train0, X_test0, y_train0, y_test0 = train_test_split(slike_0, labele_0, random_state=104, train_size=0.7,
                                                            shuffle=True)

    X_test0, X_val0, y_test0, y_val0 = train_test_split(X_test0, y_test0, random_state=104, train_size=0.5,
                                                        shuffle=True)
    # 1
    X_train1, X_test1, y_train1, y_test1 = train_test_split(slike_1, labele_1, random_state=104, train_size=0.7,
                                                            shuffle=True)

    X_test1, X_val1, y_test1, y_val1 = train_test_split(X_test1, y_test1, random_state=104, train_size=0.5,
                                                        shuffle=True)
    # 2
    X_train2, X_test2, y_train2, y_test2 = train_test_split(slike_2, labele_2, random_state=104, train_size=0.7,
                                                            shuffle=True)

    X_test2, X_val2, y_test2, y_val2 = train_test_split(X_test2, y_test2, random_state=104, train_size=0.5,
                                                        shuffle=True)
    # 3
    X_train3, X_test3, y_train3, y_test3 = train_test_split(slike_3, labele_3, random_state=104, train_size=0.7,
                                                            shuffle=True)

    X_test3, X_val3, y_test3, y_val3 = train_test_split(X_test3, y_test3, random_state=104, train_size=0.5,
                                                        shuffle=True)

    trening_slike = np.concatenate((X_train0, X_train1, X_train2, X_train3))
    val_slike = np.concatenate((X_val0, X_val1, X_val2, X_val3))
    test_slike = np.concatenate((X_test0, X_test1, X_test2, X_test3))

    trening_labele = np.concatenate((y_train0, y_train1, y_train2, y_train3))
    val_labele = np.concatenate((y_val0, y_val1, y_val2, y_val3))
    test_labele = np.concatenate((y_test0, y_test1, y_test2, y_test3))

    # slike_ulaz = np.concatenate((slike_0, slike_1, slike_2, slike_3))
    # labele_ulaz = np.concatenate((labele_0, labele_1, labele_2, labele_3))
    # print(slike_ulaz)
    # print(labele_ulaz)

    trening_labele_matrica = to_categorical(trening_labele)
    val_labele_matrica = to_categorical(val_labele)
    test_labele_matrica = to_categorical(test_labele)
    # labele_matrica = to_categorical(labele_ulaz)
    # X_train, X_test, y_train, y_test = train_test_split(slike_ulaz, labele_matrica, random_state=104, train_size=0.7,
    #                                                     shuffle=True)
    #
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=104, train_size=0.5,
    #                                                 shuffle=True)

    # # TODO: proveri koliko kojih podataka je u kom skupu
    #
    # grid search
    # dropout_rate = [0.1, 0.2, 0.3]
    # batch_size = [8, 32, 64]
    # param_grid = dict(batch_size=batch_size)
    # model = KerasClassifier(model=pretrenirani_model)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(trening_slike, trening_labele_matrica, batch=5)
    #
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    #checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    #optimalno za dropout 0.3
    # model = pretrenirani_model()
    # early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    # callbacks_list = [checkpoint, early]  # early

    # history = model.fit(X_train, y_train, batch_size=10, epochs=30, shuffle=True, verbose=True, callbacks=[early],
    #                     validation_data=(X_val, y_val))

    # history = model.fit(trening_slike, trening_labele_matrica, epochs=20, shuffle=True, verbose=True,
    #                     callbacks=[early],
    #                     validation_data=(val_slike, val_labele_matrica))

    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    #cuvanje modela
    # model.save(rf"model_undersampling_dropout.h5")
    model = keras.saving.legacy.save.load_model("model_skalirano.h5")

    # matrica konfuzije za test
    predikcije_test = model.predict(test_slike, verbose=True)
    predikcije_test = np.argmax(predikcije_test, axis=1)
    y_test = np.argmax(test_labele_matrica, axis=1)

    # cm_test = confusion_matrix(y_test, predikcije_test, normalize='true')
    # cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1, 2, 3])
    # cmDisplay.plot()
    # plt.show()

    # matrica konfuzije za trening
    predikcije_trening = model.predict(trening_slike, verbose=True)
    predikcije_trening = np.argmax(predikcije_trening, axis=1)
    y_trening = np.argmax(trening_labele_matrica, axis=1)

    # cm_trening = confusion_matrix(y_trening, predikcije_trening, normalize='true')
    # cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm_trening, display_labels=[0, 1, 2, 3])
    # cmDisplay.plot()
    # plt.show()

    misclassified_indices = np.where(y_test != predikcije_test)[0]
    misclassified_df = pd.DataFrame({'true_label': y_test[misclassified_indices],
                                     'predicted_label': predikcije_test[misclassified_indices]},
                                    index=misclassified_indices)
    print(misclassified_df)
    misclassified_images = test_slike[misclassified_indices]
    for i in range(max(test_slike.shape)):
        cv2.imshow(f'Predicted: {predikcije_test[misclassified_indices[i]]}',misclassified_images[i])
        cv2.waitKey(0)
    #todo: tamo gde se slika razlikuje od labele to je lose klasifikovano
