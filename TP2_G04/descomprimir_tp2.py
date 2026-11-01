#!/usr/bin/env python3
"""Descomprime un archivo .tp2 y muestra la imagen.

Uso: python descomprimir_tp2.py archivo.tp2 [-o imagen.png]
"""
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from formato_tp2 import descomprimir


def main():
    ap = argparse.ArgumentParser(description='Descomprime un .tp2 y muestra la imagen.')
    ap.add_argument('archivo', help='archivo .tp2 de entrada')
    ap.add_argument('-o', '--salida', help='guardar además la imagen descomprimida (p. ej. salida.png)')
    args = ap.parse_args()

    with open(args.archivo, 'rb') as f:
        img = descomprimir(f.read())

    print(f'{args.archivo} -> imagen {img.shape[1]}x{img.shape[0]} (grises)')
    if args.salida:
        Image.fromarray(img).save(args.salida)
        print(f'Guardada en {args.salida}')

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(args.archivo)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
