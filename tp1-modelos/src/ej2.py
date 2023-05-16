#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle

import common


META = dict(
    # distribucion de nodos por capa
    structure=[8, 16, 8, 2],
    # lr de la red
    learning_rate=0.3,
    # adaptive lr
    # alfa que se le suma luego de K "pasos" en la dirección correcta
    # (es decir, si W decremento el error)
    learning_rate_a=1e-2,
    # beta que indica cuanto decrementar lr cuando se da un mal paso (W
    # incremento el error)
    learning_rate_b=1e-2,
    # cantidad de pasos buenos consecutivos que se necesitan para aumentar lr
    learning_rate_k=3,
    # cantidad de épocas del entrenamiento
    epochs=2000,
    # momento (incrementa el deltaW en la dirección del deltaW anterior)
    moment=0.9,
    # beta para la sigmoidea
    sigmoid_beta=1,
    # porcentaje de los datos reservados para test
    perc_test=0.10,
    # reservado para entrenamiento es lo que queda (P-P_test)
    # perc_train=0.9

    # margen de error: Porcentaje que define que predicciones son correctas o
    # no. Si la prediccion es Y, y la respuesta esperada era Z, consideramos Y
    # correcta si abs(Z - Y) < Z*error_margin
    error_margin=0.05,

    # tamaño de los lotes de entrenamiento
    batch_size=8
)


cwd = os.getcwd()
join = os.path.join
isfile = os.path.isfile


def cargar_datos(f):
    df = np.loadtxt(f, delimiter=',')
    # ponemos todos los datos de entrada en X
    X = df[:, :-2]
    # y los de salida en Z
    Z = df[:, -2:]

    return (normalizar(X), normalizar(Z), Z)


def normalizar(a):
    return (a - a.mean(0)) / a.std(0, ddof=1)


def desnormalizar(a, mean, std):
    s = a.shape
    mean = np.repeat(mean.reshape(1, s[1]), s[0], axis=0)
    std = np.repeat(std.reshape(1, s[1]), s[0], axis=0)
    return np.multiply(a,  std) + mean


def estructura_de_pesos(W):
    S = []
    for w in W:
        S.append(w.shape[0]-1)
    S.append(W[-1].shape[1])

    return S


def usage():
    print(
        "usage: ./ej2.py MODELO DATOS\n\n"
        "si MODELO existe, se cargara el mismo y se usara DATOS para testeo\n"
        "si MODELO no existe, se entrenara un nuevo modelo utilizando DATOS y"
        "se guardará en MODELO")
    sys.exit(1)


def test(W, X, Z, Zo, mean=None, std=None):
    """Devuelve el porcentaje de resultados que la red predijo correctamente

    Obtiene la respuesta de la red, dados los parámetros W, X, y la respuesta
    esperada Z (normalizada), y los desnormaliza para compararlos con la
    respuesta real.

    Sea C_e la cantidad de valores que la red devolvió que se encuentran en un
    margen de +-(100*error_margin)% del valor esperado, y C el total de valores
    que devolvió (cantidad de casos de test * cantidad de unidades de salida).
    Entonces esta función devuelve:
        C_e / C

    Esto quiere decir que si la red predijo correctamente un valor de salida,
    pero no el otro, de todas formas aporta a la precisión.

    Se puede cambiar utilizando la alternativa debajo del return
    """
    mean = mean if mean is not None else Zo.mean(0)
    std = std if std is not None else Zo.std(0, ddof=1)
    e = META["error_margin"]
    Y = common.activacion(X, W, **META)
    Y = desnormalizar(Y[-1], mean, std)
    ok = 0
    V = np.abs(Zo - Y) / Zo

    V = (V < e).flatten()
    for v in V:
        if v:
            ok += 1
    return ok / len(V)

    # ALTERNATIVA
    # considerar como predicción correcta los casos en los que ambos valores
    # se encuentran en el margen de error
    # V = V < e
    # for z0, z1 in V:
    #     if z0 and z1:
    #         ok += 1
    # return ok / len(V)


def main():
    if len(sys.argv) != 3:
        usage()
        return

    modelo = sys.argv[1]
    datos = sys.argv[2]

    # archivo de datos tiene que existir
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
        X, Z, Zo = cargar_datos(datos)

        acc = test(W, X, Z, Zo) * 100
        print(f">> Casos identificados correctamente: {acc}%")
    else:
        print("- Cargando datos")
        X, Z, Zo = cargar_datos(datos)

        P = len(X)
        p_test = META["perc_test"]
        c_test = int(P * p_test)
        r_training = range(0, P - c_test)
        print(f"- Usando rango {r_training} para entrenamiento")

        W = common.inicializar_pesos(**META)
        W = common.entrenamiento(W, X[r_training], Z[r_training], **META)

        r_test = range(P - c_test, P)
        mean = Zo.mean(0)
        std = Zo.std(0, ddof=1)
        print(f"- Usando rango {r_test} para testear el modelo obtenido")
        acc = test(W, X[r_test], Z[r_test], Zo[r_test], mean, std) * 100
        e = META["error_margin"]
        print(f">> Valores obtenidos dentro del margen de error ({e}): {acc}%")

        print(f"- Guardando modelo generado en '{modelo}'")
        pickle.dump(W, fmodelo)

    if fmodelo:
        fmodelo.close()


if __name__ == '__main__':
    main()
