#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import pickle

import common


META = dict(
    # distribucion de nodos por capa
    structure=[10, 20, 8, 1],
    # lr de la red
    learning_rate=1e-2,
    # epocas
    epochs=1000,
    # porcentaje de los datos reservados para test
    perc_test=0.05,
    # reservado para validacion
    perc_val=0.15,
    # reservado para entrenamiento es lo que queda (P-P_test-P_val)
    # perc_train=0.8

    # tamaño de los lotes de entrenamiento
    batch_size=16
)


def cargar_datos(f):
    path = os.path.join(os.getcwd(), f)
    df = pd.read_csv(path, header=None)
    # ponemos todos los datos de entrada en X
    X = df.loc[:, 1:].to_numpy()
    # los normalizamos
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    # y los de salida en Z
    Z = np.array([1 if d == 'B' else -1 for d in df.loc[:, 0]])

    return (X, Z.reshape(len(Z), 1))


def estructura_de_pesos(W):
    S = []
    for w in W:
        S.append(w.shape[0]-1)
    S.append(W[-1].shape[1])

    return S


def usage():
    print(
        "usage: ./ej1.py MODELO DATOS\n\n"
        "si MODELO existe, se cargara el mismo y se usara DATOS para testeo\n"
        "si MODELO no existe, se entrenara un nuevo modelo utilizando DATOS y"
        "se guardará en MODELO")
    sys.exit(1)


def test(W, X, Z):
    Y = common.activacion(X, W, **META)
    Y = np.sign(Y[-1])
    ok = 0
    V = Z - Y
    for v in V:
        if v == 0:
            ok += 1
    return ok / len(V)


cwd = os.getcwd()
join = os.path.join
isfile = os.path.isfile


def main():
    if len(sys.argv) != 3:
        usage()
        return

    modelo = sys.argv[1]
    datos = sys.argv[2]

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

    if modelo_existe:
        global META

        print("- Cargando modelo en W")
        W = pickle.load(fmodelo)
        # modificar estructura para que sea compatible con el W cargado
        META["structure"] = estructura_de_pesos(W)

        print("- Cargando los datos de prueba")
        X, Z = cargar_datos(datos)

        acc = test(W, X, Z) * 100
        print(f"* Casos identificados correctamente: {acc}%")
    else:
        print("- Cargando datos")
        X, Z = cargar_datos(datos)

        P = len(X)
        p_test = META["perc_test"]
        c_test = int(P * p_test)
        r_training = range(0, P - c_test)
        print(f"- Usando rango {r_training} para entrenamiento")

        W = common.inicializar_pesos(**META)
        W = common.entrenamiento(W, X[r_training], Z[r_training], **META)
        r_test = range(P - c_test, P)
        print(f"- Usando rango {r_test} para testear el modelo obtenido")

        acc = test(W, X[r_test], Z[r_test]) * 100
        print(f"* Casos identificados correctamente: {acc}%")

        print(f"- Guardando modelo generado en '{modelo}'")
        pickle.dump(W, fmodelo)

    if fmodelo:
        fmodelo.close()


if __name__ == '__main__':
    main()
