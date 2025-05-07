import sys
import cv2
from services.ocr_cluster import analyze_and_cluster

def to_px(val, dim):
    if 0 <= val <= 1:
        return int(val * dim)
    return int(val)

def debug_image(image_path: str):
    """
    Executa a análise e clusterização de OCR na imagem original,
    desenha as bounding boxes e imprime coordenadas.
    """
    # 1️⃣ Chama o módulo para extrair dados
    result = analyze_and_cluster(image_path)
    print(result)

    # 2️⃣ Carrega a imagem ORIGINAL (sem resize)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: não foi possível carregar '{image_path}'")
        return
    h, w = img.shape[:2]
    print(f"Imagem carregada no tamanho original: {w}x{h}")

    # 3️⃣ Define espessuras de linha proporcionais ao tamanho da imagem
    base = min(w, h)
    thick_main = max(2, int(base * 0.01))
    thick_inner = max(1, int(base * 0.005))

    # 4️⃣ Desenha bounding boxes internas
    for line in result['cluster_lines']:
        xs = [pt.x for pt in line['bbox']]
        ys = [pt.y for pt in line['bbox']]
        x1, y1 = to_px(min(xs), w), to_px(min(ys), h)
        x2, y2 = to_px(max(xs), w), to_px(max(ys), h)
        print(f"Linha: '{line['text']}', bbox pixels: ({x1},{y1})->({x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (144, 238, 144), thick_inner)

    # 5️⃣ Desenha bounding box principal
    bbox = result.get('bbox')
    if bbox:
        x1, y1 = to_px(bbox['xmin'], w), to_px(bbox['ymin'], h)
        x2, y2 = to_px(bbox['xmax'], w), to_px(bbox['ymax'], h)
        print(f"Cluster bbox principal pixels: ({x1},{y1})->({x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick_main)

    # 6️⃣ Exibe a imagem final
    cv2.namedWindow('Debug OCR Cluster', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug OCR Cluster', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Uso: python debug_image.py <caminho_da_imagem>')
    else:
        debug_image(sys.argv[1])
