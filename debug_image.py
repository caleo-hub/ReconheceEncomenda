import cv2
from services.ocr_cluster import analyze_and_cluster
from services.barcode_reader import ler_codigos_barra_e_qr

import logging, sys

def setup_module_logger(name: str, level=logging.INFO) -> logging.Logger:
    # 1) silencia o root para WARNING+ de tudo
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for h in list(root.handlers):
        root.removeHandler(h)

    # 2) handler para o seu módulo
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    # 3) configura o logger do módulo
    lm = logging.getLogger(name)
    lm.setLevel(level)
    lm.propagate = False
    lm.addHandler(handler)
    return lm

# configurações de log
# seu próprio logger (debug_image.py)
logger = setup_module_logger(__name__, level=logging.INFO)
# e o do ocr_cluster
ocr_logger = setup_module_logger("services.ocr_cluster", level=logging.DEBUG)

def to_px(val, dim):
    if 0 <= val <= 1:
        return int(val * dim)
    return int(val)

def debug_image(image_path: str):
    """
    Executa a análise e clusterização de OCR na imagem original,
    desenha as bounding boxes e imprime coordenadas.
    """
    # 1 Chama o módulo para extrair dados
    result = analyze_and_cluster(image_path)


    # 2 Carrega a imagem ORIGINAL (sem resize)
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Erro: não foi possível carregar '{image_path}'")
        return
    h, w = img.shape[:2]
    logger.info(f"Imagem carregada no tamanho original: {w}x{h}")

    # 3 Define espessuras de linha proporcionais ao tamanho da imagem
    base = min(w, h)
    thick_main = max(2, int(base * 0.01))
    thick_inner = max(1, int(base * 0.005))

    # 4 Desenha as bounding boxes do destinatario e CEP
    for key in ['recipient', 'cep']:
        bbox = result.get(key, {}).get('bbox')
        if bbox:
            xs = [pt.x for pt in bbox]
            ys = [pt.y for pt in bbox]
            x1, y1 = to_px(min(xs), w), to_px(min(ys), h)
            x2, y2 = to_px(max(xs), w), to_px(max(ys), h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thick_main)
    
    # 5 Desenha bounding boxes internas
    for line in result['cluster_lines']:
        xs = [pt.x for pt in line['bbox']]
        ys = [pt.y for pt in line['bbox']]
        x1, y1 = to_px(min(xs), w), to_px(min(ys), h)
        x2, y2 = to_px(max(xs), w), to_px(max(ys), h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (192, 192, 192), thick_inner)

    # 6 Desenha bounding box principal
    bbox = result.get('bbox')
    if bbox:
        x1, y1 = to_px(bbox['xmin'], w), to_px(bbox['ymin'], h)
        x2, y2 = to_px(bbox['xmax'], w), to_px(bbox['ymax'], h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick_main)

    
    # 7 Exibe a imagem final
    cv2.namedWindow('Debug OCR Cluster', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug OCR Cluster', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Uso: python debug_image.py <caminho_da_imagem>')
    else:
        debug_image(sys.argv[1])

        
