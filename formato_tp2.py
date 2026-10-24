"""
Formato binario .tp2: imagen en escala de grises comprimida con Huffman sobre residuos por fila.

Estructura del archivo (enteros en little-endian):
    magic         4 bytes   b'TP2\\x01'
    M, N          2 x uint32  alto y ancho de la imagen
    n_simbolos    uint16    cantidad de residuos distintos (0 si N == 1)
    tabla         por simbolo: residuo int16 | largo del codigo uint16 | codigo empaquetado en bytes
    primera_col   M bytes   primer pixel de cada fila (condicion inicial)
    n_bits        uint64    cantidad de bits utiles del payload
    payload       bits de los M*(N-1) residuos, empaquetados (relleno de ceros al final)
"""
import struct
import numpy as np

from funciones_auxiliares_tp2 import huffman_code, huffman_decode

MAGIC = b'TP2\x01'


def _pmf(simbolos):
    valores, cuentas = np.unique(simbolos, return_counts=True)
    return {int(v): c / simbolos.size for v, c in zip(valores, cuentas)}


def _empaquetar(bits):
    """String de '0'/'1' -> bytes (relleno de ceros a la derecha)."""
    if not bits:
        return b''
    arr = np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - 48
    return np.packbits(arr).tobytes()


def _desempaquetar(datos, n_bits):
    """bytes -> string de '0'/'1' de largo n_bits."""
    if n_bits == 0:
        return ''
    arr = np.unpackbits(np.frombuffer(datos, dtype=np.uint8))[:n_bits]
    return (arr + 48).astype(np.uint8).tobytes().decode('ascii')


def comprimir(img):
    """img: matriz 2D uint8/int. Retorna los bytes del archivo .tp2."""
    img = np.asarray(img, dtype=np.int16)
    M, N = img.shape
    primera_col = img[:, 0].astype(np.uint8)
    res = np.diff(img, axis=1)          # (M, N-1), residuos entre vecinos de una misma fila

    salida = [MAGIC, struct.pack('<IIH', M, N, 0)]
    codigo = {}
    bits = ''
    if res.size > 0:
        codigo = huffman_code(_pmf(res.ravel()))
        if len(codigo) == 1:            # un unico residuo: huffman_code devuelve '' -> forzar 1 bit
            codigo = {s: '0' for s in codigo}
        bits = ''.join(codigo[int(r)] for r in res.ravel())
    salida[1] = struct.pack('<IIH', M, N, len(codigo))

    for simbolo, cod in codigo.items():
        salida.append(struct.pack('<hH', simbolo, len(cod)))
        salida.append(_empaquetar(cod))
    salida.append(primera_col.tobytes())
    salida.append(struct.pack('<Q', len(bits)))
    salida.append(_empaquetar(bits))
    return b''.join(salida)


def descomprimir(datos):
    """bytes de un .tp2 -> matriz uint8 (M, N)."""
    if datos[:4] != MAGIC:
        raise ValueError('No es un archivo .tp2 valido (magic incorrecto).')
    pos = 4
    M, N, n_simbolos = struct.unpack_from('<IIH', datos, pos); pos += 10

    codigo = {}
    for _ in range(n_simbolos):
        simbolo, largo = struct.unpack_from('<hH', datos, pos); pos += 4
        nbytes = (largo + 7) // 8
        codigo[simbolo] = _desempaquetar(datos[pos:pos + nbytes], largo); pos += nbytes

    primera_col = np.frombuffer(datos, dtype=np.uint8, count=M, offset=pos).astype(np.int16); pos += M
    (n_bits,) = struct.unpack_from('<Q', datos, pos); pos += 8
    bits = _desempaquetar(datos[pos:], n_bits)

    if N == 1:
        res = np.zeros((M, 0), dtype=np.int16)
    else:
        res = np.array(huffman_decode(bits, codigo), dtype=np.int16).reshape(M, N - 1)
    img = np.concatenate([primera_col[:, None], primera_col[:, None] + np.cumsum(res, axis=1)], axis=1)
    return img.astype(np.uint8)
