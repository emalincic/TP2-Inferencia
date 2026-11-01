
"""Comprime una imagen (jpg, png, ...) a un archivo .tp2 (Huffman sobre residuos).
"""
import argparse
import os
import numpy as np
from PIL import Image

from formato_tp2 import comprimir


def main():
    ap = argparse.ArgumentParser(description='Compresión sin pérdida con Huffman sobre residuos.')
    ap.add_argument('imagen', help='imagen de entrada (jpg, png, ...)')
    ap.add_argument('-o', '--salida', help='archivo de salida (por defecto: mismo nombre con extensión .tp2)')
    args = ap.parse_args()

    img = np.array(Image.open(args.imagen).convert('L'))
    datos = comprimir(img)

    salida = args.salida or os.path.splitext(args.imagen)[0] + '.tp2'
    with open(salida, 'wb') as f:
        f.write(datos)

    bits_px = 8 * len(datos) / img.size
    print(f'{args.imagen} ({img.shape[1]}x{img.shape[0]}, grises) -> {salida}')
    print(f'{len(datos)} bytes, {bits_px:.3f} bits/píxel, razón de compresión {8 / bits_px:.3f}x respecto de 8 bits/píxel')


if __name__ == '__main__':
    main()
