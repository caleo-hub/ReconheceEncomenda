import re
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from typing import Optional, Dict, List


def extrair_cep(texto: str) -> Optional[str]:
    match = re.search(r'\b\d{5}-?\d{3}\b', texto)
    return match.group(0) if match else None


def extrair_codigo_rastreio(texto: str) -> Optional[str]:
    match = re.search(r'\b[A-Z]{2}\d{9}[A-Z]{2}\b', texto, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _tentar_decodificar_variacoes(imagem: np.ndarray):
    """
    Tenta várias formas de preprocessamento para aumentar a chance de leitura dos códigos.
    """
    tentativas = []

    # Original
    tentativas.append(imagem)

    # Tons de cinza com threshold adaptativo
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    tentativas.append(thresh)

    # Invertida (para códigos escuros em fundo claro)
    invertida = cv2.bitwise_not(thresh)
    tentativas.append(invertida)

    # Tentar decodificar em cada uma
    for tentativa in tentativas:
        codigos = decode(tentativa)
        if codigos:
            return codigos
    return []


def ler_codigos_barra_e_qr(image_path: str) -> Dict[str, Optional[str] | List[str]]:
    """
    Lê todos os códigos de barras e QR codes presentes na imagem e retorna
    o CEP, o código de rastreio e outros códigos lidos.

    :param image_path: Caminho para a imagem
    :return: Dicionário com as chaves 'cep', 'codigo_rastreio' e 'outros_codigos'
    """
    imagem = cv2.imread(image_path)
    if imagem is None:
        raise ValueError(f"Imagem não encontrada ou inválida: {image_path}")

    codigos_detectados = _tentar_decodificar_variacoes(imagem)

    cep = None
    codigo_rastreio = None
    outros_codigos = []

    for codigo in codigos_detectados:
        texto = codigo.data.decode("utf-8").strip()
        outros_codigos.append(texto)
        if not cep:
            cep = extrair_cep(texto)
        if not codigo_rastreio:
            codigo_rastreio = extrair_codigo_rastreio(texto)

    return {
        "cep": cep,
        "codigo_rastreio": codigo_rastreio,
        "outros_codigos": outros_codigos
    }
