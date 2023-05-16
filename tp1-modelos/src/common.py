import numpy as np
from matplotlib import pyplot as plt


def _agregar_bias(a):
    return np.concatenate((a, -np.ones((len(a), 1))), axis=1)


def _quitar_bias(a):
    return a[:, :-1]


def _get_or_error(d, k):
    if k not in d:
        raise KeyError(f'{k} no esta definido en el diccionario')

    return d[k]


def _get_or_default(d, k, default):
    return default if k not in d else d[k]


def inicializar_pesos(**kwargs):
    """Inicializa los pesos con valores del orden del fan-in de cada capa"""
    S = _get_or_error(kwargs, "structure")
    W = [np.random.randn(S[i]+1, S[i+1]) * 1/np.sqrt(S[i]) for i in range(len(S)-1)]
    return W


def activacion(X, W, **kwargs):
    """Propaga los datos de entrada por la red hacia adelante"""
    S = _get_or_error(kwargs, "structure")
    B = _get_or_default(kwargs, "sigmoid_beta", 1)

    # current batch size
    b = len(X)
    # activacion de cada nodo (para cada patron del batch)
    Y = [np.empty((b, s+1)) for s in S]
    # la ultima capa no tiene nodo de bias
    Y[-1] = np.empty((b, S[-1]))
    # Y sin bias
    Ysb = X

    for i in range(1, len(S)):
        Y[i-1][:, :] = _agregar_bias(Ysb)
        Ysb = np.tanh(B * np.dot(Y[i-1], W[i-1]))

    Y[-1][:, :] = Ysb

    return Y


def correccion(W, Y, Z, **kwargs):
    """Genera delta W haciendo back-propagation"""
    LR = _get_or_error(kwargs, "learning_rate")
    B = _get_or_default(kwargs, "sigmoid_beta", 1)
    M = _get_or_default(kwargs, "moment", 0)
    pdW = _get_or_default(kwargs, "prev_dW", None)

    # tamaño del lote actual
    b = len(Z)
    # delta W que se le sumará a W
    dW = [np.empty_like(w) for w in W]
    # delta de cada capa
    D = [np.empty_like(y) for y in Y]
    # derivada de la sigmoidea (tanh en nuestro caso)
    dY = B * (1 - np.square(Y[-1]))
    D[-1] = np.multiply(dY, Z - Y[-1])

    for k in range(len(Y)-1, 0, -1):
        o = [np.outer(Y[k-1][i], D[k][i]) for i in range(b)]
        dW[k-1] = LR * np.mean(o, axis=0)
        if M and pdW:
            dW[k-1] += M * pdW[k-1]
        p = np.dot(D[k], W[k-1].T)
        dY = B * (1 - np.square(Y[k-1]))
        D[k-1] = _quitar_bias(np.multiply(p, dY))

    return dW


def adaptacion(W, dW, **kwargs):
    """Aplica los dW a cada capa de W"""

    return [w + dW[i] for i, w in enumerate(W)]


def entrenamiento(W, X, Z, **kwargs):
    T = _get_or_error(kwargs, "epochs")
    B = _get_or_error(kwargs, "batch_size")
    LR = _get_or_error(kwargs, "learning_rate")
    LR_A = _get_or_default(kwargs, "learning_rate_a", 0)
    LR_B = _get_or_default(kwargs, "learning_rate_b", 0)
    LR_K = _get_or_default(kwargs, "learning_rate_k", 0)
    adaptive_lr = LR_K != 0

    # epoca actual
    t = 0
    # cantidad de "pasos" en la "direccion correcta"
    k = 0

    error_values = []

    dW = None  # ultimo dW
    E_last = None  # ultimo error
    while t < T:
        e = 0
        indices = np.random.permutation(len(X))
        for batch in range(0, len(indices), B):
            h = indices[batch:batch+B]
            Xh = X[h]
            Zh = Z[h]
            Yh = activacion(Xh, W, **kwargs)

            if adaptive_lr:
                E = np.square(Zh - Yh[-1]).sum() / 2
                dE = E - E_last if E_last is not None else 0
                E_last = E

                # delta E > 0: error aumentó
                if dE > 0:
                    LR -= LR_B * LR  # disminuir el LR
                    k = 0  # resetear contador de pasos correctos
                    # deshacer ultimo paso
                    W = adaptacion(W, [-dw for dw in dW])
                    kwargs["learning_rate"] = LR
                    # delta E: < 0: error disminuyo
                elif dE < 0:
                    k += 1
                    if k == LR_K:
                        LR += LR_A
                        kwargs["learning_rate"] = LR
                        k -= 1  # perminir varios incrementos consecutivos

            e += np.mean(np.square(Zh - Yh[-1]))
            # e += np.mean(np.square(np.subtract(Zh, Yh[L-1])))
            dW = correccion(W, Yh, Zh, prev_dW=dW, **kwargs)
            W = adaptacion(W, dW)
        error_values.append(e)
        if t % 100 == 0:
            print(f"#{t}\terror = {e:.3f}\tlr={LR:.5f}")
        t += 1

    plt.plot(error_values, 'r')
    plt.xlabel("epoca")
    plt.ylabel("error")
    plt.show()

    return W
