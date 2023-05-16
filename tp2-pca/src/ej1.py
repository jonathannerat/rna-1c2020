#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import pickle


rs = np.random.RandomState(seed=1992340)
cwd = os.getcwd()
join = os.path.join
isfile = os.path.isfile

P = N = M = None


def cargar_datos(path):
    data = np.loadtxt(path, delimiter=',')

    # separamos categorias asignadas de cada instancia
    CAT = data[:, 0]
    X = data[:, 1:]
    return (CAT, ((X - X.mean(axis=0)) / X.std(axis=0, ddof=1)))


def inicializar_pesos():
    return rs.normal(0, 0.1, (N, M))


def ortogonalidad(W):
    return np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2


def correccion_oja(W, X, Y):
    Z = np.dot(Y, W.T)
    return np.outer(X-Z, Y)


def correccion_sanger(W, X, Y):
    D = np.triu(np.ones((M, M)))
    Z = np.dot(W, Y.T*D)

    return (X.T-Z)*Y


def entrenamiento(W, X, corr):
    MAX_EPOCAS = 400
    MAX_ORT = 0.05
    LR = 1e-3
    t = 0 # epoca
    ort = 1
    hort = []
    lr = LR
    lra = 1

    while ort > MAX_ORT and t < MAX_EPOCAS:
        for i in range(P):
            x = X[i:i+1] # para forzar un shape (1, len(x))
            Y = np.dot(x, W)
            dW = lr * corr(W, x, Y)
            W += dW

        ort = ortogonalidad(W)
        hort.append(ort)
        if t % 10 == 0:
            print(f"    epoca: {t:03} | ortog: {ort}")
        t += 1
        lra += 0.6
        lr = LR / lra
    plt.plot(hort)
    plt.title("Ortogonalidad de W por epoca")
    plt.xlabel("t")
    plt.ylabel("ort(W)")
    plt.show()
    return W


def visualizar(W, X, CAT):
    Y = np.dot(X, W)  # respuesta del modelo

    for i in range(3):
        fig = plt.figure()
        xyz = fig.add_subplot(111, projection="3d")
        xyz.set_xlim(-10, 10)
        xyz.set_ylim(-10, 10)
        xyz.set_zlim(-10, 10)
        xyz.scatter3D(Y[:, 3*i+0], Y[:, 3*i+1], Y[:, 2], c=CAT, cmap="Dark2", alpha=True)
        xyz.set_title(f"Subespacio definido por ejes Y{3*i+1}, Y{3*i+2}, Y{3*i+3}")
        xyz.set_xlabel(f"Y{3*i+0}")
        xyz.set_ylabel(f"Y{3*i+1}")
        xyz.set_zlabel(f"Y{3*i+2}")
        fig.show()
    plt.show()


def usage():
    print(
        "usage: ./ej1.py [--oja|--sanger] MODELO DATOS\n\n"
        "opciones:\n"
        "   [--oja | --sanger]  Parametro opcional que indica que regla de \n"
        "                       corrección se utilizará para entrenar el modelo en\n"
        "                       caso de que MODELO no exista. Default: sanger\n"
        "   MODELO              Si existe, se cargara el mismo y se usara DATOS para\n"
        "                       testeo. Caso contrario, se entrenará un nuevo modelo\n"
        "                       utilizando DATOS y se guardara en MODELO.\n"
        "   DATOS               Datos de entrada para entrenamiento / testeo\n")
    sys.exit(1)


def main():
    global P, N, M

    argc = len(sys.argv)
    dio_correccion = argc == 4 and sys.argv[1] in ["--oja", "--sanger"]
    if argc < 3 or (not dio_correccion and argc > 3):
        usage()
        return

    # default
    corr = correccion_sanger
    modelo = sys.argv[1]
    datos = sys.argv[2]
    usa_sanger = True

    if dio_correccion:
        if sys.argv[1] == "--oja":
            corr = correccion_oja
            usa_sanger = False
        modelo = sys.argv[2]
        datos = sys.argv[3]

    if not isfile(join(cwd, datos)):
        usage()
        return

    modelo_existe = False
    fmodelo = None

    if isfile(join(cwd, modelo)):
        fmodelo = open(modelo, "rb")
        modelo_existe = True
    else:
        fmodelo = open(modelo, "wb")

    CAT, X = cargar_datos(datos)
    P = X.shape[0]
    N = X.shape[1]
    M = 9

    print(CAT.shape, X.shape)

    if modelo_existe:
        print("- Cargando los datos de prueba")

        print("- Cargando modelo en W")
        W = pickle.load(fmodelo)

        print("- Visualizacion de datos")
        visualizar(W, X, CAT)
    else:
        print("- Cargando datos para entrenamiento")

        W = inicializar_pesos()

        corr_elegida = "Sanger" if usa_sanger else "Oja"
        print(f"- Entrenando modelo con regla de {corr_elegida}")
        W = entrenamiento(W, X, corr)

        print(f"- Guardando modelo entrenado en {modelo}")
        pickle.dump(W, fmodelo)

    if fmodelo:
        fmodelo.close()


if __name__ == '__main__':
    main()
