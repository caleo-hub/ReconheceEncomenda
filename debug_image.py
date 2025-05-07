# import os
# import re
# import cv2
# import sys
# import numpy as np
# from dotenv import load_dotenv
# from sklearn.cluster import DBSCAN
# from azure.ai.vision.imageanalysis import ImageAnalysisClient
# from azure.ai.vision.imageanalysis.models import VisualFeatures
# from azure.ai.textanalytics import TextAnalyticsClient
# from azure.core.credentials import AzureKeyCredential

# # Carrega variáveis de ambiente necessárias para autenticação Azure
# load_dotenv()
# API_KEY = os.getenv("MULTISERVICE_API_KEY")
# ENDPOINT = os.getenv("MULTISERVICE_ENDPOINT")
# if not API_KEY or not ENDPOINT:
#     raise EnvironmentError("Defina MULTISERVICE_API_KEY e MULTISERVICE_ENDPOINT no .env")

# # Inicializa clientes Azure Vision e Text Analytics
# vision_client = ImageAnalysisClient(
#     endpoint=ENDPOINT,
#     credential=AzureKeyCredential(API_KEY)
# )
# text_client = TextAnalyticsClient(
#     endpoint=ENDPOINT,
#     credential=AzureKeyCredential(API_KEY)
# )

# # Função auxiliar: converte valor normalizado (0-1) ou absoluto em pixels
# def to_px(val, dim):
#     return int(val * dim) if 0 <= val <= 1 else int(val)


# def process_image_with_debug(image_path):
#     """
#     Carrega uma imagem, executa OCR + NER, clusteriza em torno de destinatário/CEP,
#     desenha retângulos e exibe em janela redimensionável com logs de coordenadas.
#     """
#     # 1️⃣ Carrega imagem do disco
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Erro: não foi possível abrir '{image_path}'")
#         return

#     # 2️⃣ Redimensiona imagem para largura fixa (mantendo proporção)
#     TARGET_W = 1000
#     h0, w0 = img.shape[:2]
#     ratio = TARGET_W / float(w0)
#     img = cv2.resize(img, (TARGET_W, int(h0 * ratio)))
#     h, w = img.shape[:2]
#     print(f"Imagem redimensionada para: {w}x{h}")

#     # 3️⃣ Define espessuras de linhas proporcionais ao tamanho da imagem
#     base = min(w, h)
#     thick_main = max(2, int(base * 0.01))    # 1% do menor lado
#     thick_inner = max(1, int(base * 0.005))  # 0.5% do menor lado

#     # 4️⃣ Salva temporariamente a imagem redimensionada para envio ao OCR
#     tmp_file = image_path + ".tmp.jpg"
#     cv2.imwrite(tmp_file, img)

#     # 5️⃣ Executa OCR com Azure Vision (VisualFeatures.READ)
#     with open(tmp_file, 'rb') as f:
#         ocr_res = vision_client.analyze(
#             image_data=f,
#             visual_features=[VisualFeatures.READ],
#             language='pt'
#         )
#     os.remove(tmp_file)

#     # 6️⃣ Extrai todas as linhas do OCR, calculando bounding box e centroide
#     lines = []
#     if ocr_res.read and ocr_res.read.blocks:
#         for block in ocr_res.read.blocks:
#             for ln in block.lines:
#                 txt = ln.text.strip()
#                 bbox = ln.bounding_polygon  # 4 pontos com x,y normalizados
#                 xs = [pt.x for pt in bbox]
#                 ys = [pt.y for pt in bbox]
#                 lines.append({
#                     'text': txt,
#                     'bbox': bbox,
#                     'cx': float(np.mean(xs)),  # centro X normalizado
#                     'cy': float(np.mean(ys))   # centro Y normalizado
#                 })
#     if not lines:
#         print("Nenhuma linha OCR detectada.")
#         return

#     # 7️⃣ Realiza NER para encontrar índice do destinatário (categoria Person)
#     full_text = "\n".join([l['text'] for l in lines])
#     ner = text_client.recognize_entities([full_text], language='pt')[0]
#     dest_idx = None
#     for ent in ner.entities:
#         if ent.category == 'Person':
#             for i, l in enumerate(lines):
#                 if ent.text in l['text']:
#                     dest_idx = i
#                     break
#         if dest_idx is not None:
#             break

#     # 8️⃣ Procura todos CEPs via regex e seleciona o mais próximo do destinatário
#     ceps = [(i, re.search(r"\b\d{5}-?\d{3}\b", l['text'])) for i, l in enumerate(lines)]
#     ceps = [(i, m.group()) for i, m in ceps if m]
#     cep_idx = None
#     if ceps:
#         if dest_idx is not None:
#             cep_idx, cep_val = min(ceps, key=lambda x: abs(x[0] - dest_idx))
#         else:
#             cep_idx, cep_val = ceps[0]
#         print(f"Encontrado CEP '{cep_val}' em linha {cep_idx}")

#     # 9️⃣ Clusterização espacial com DBSCAN usando centroides normalizados
#     coords = np.array([[l['cx'], l['cy']] for l in lines])
#     labels = DBSCAN(eps=0.01, min_samples=1).fit_predict(coords)
#     for i, l in enumerate(lines):
#         l['cluster'] = int(labels[i])

#     # 🔟 Seleciona clusters onde estão destinatário e CEP
#     relevant = set()
#     if dest_idx is not None:
#         relevant.add(lines[dest_idx]['cluster'])
#     if cep_idx is not None:
#         relevant.add(lines[cep_idx]['cluster'])
#     cluster_lines = [l for l in lines if l['cluster'] in relevant]
#     print(f"Linhas no cluster relevante: {len(cluster_lines)}")

#     # 1️⃣1️⃣ Calcula bounding box do cluster relevante (normalizado)
#     xs = [pt.x for l in cluster_lines for pt in l['bbox']]
#     ys = [pt.y for l in cluster_lines for pt in l['bbox']]
#     bbox = {
#         'xmin': min(xs), 'ymin': min(ys),
#         'xmax': max(xs), 'ymax': max(ys)
#     }

#     # 1️⃣2️⃣ Converte bbox normalizado para pixels e imprime coordenadas
#     x1, y1 = to_px(bbox['xmin'], w), to_px(bbox['ymin'], h)
#     x2, y2 = to_px(bbox['xmax'], w), to_px(bbox['ymax'], h)
#     print(f"Cluster bbox pixels: ({x1},{y1}) -> ({x2},{y2})")

#     # 1️⃣3️⃣ Desenha bounding box principal em verde
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick_main)

#     # 1️⃣4️⃣ Desenha bounding boxes internas em verde claro e imprime coords
#     for l in cluster_lines:
#         xs = [pt.x for pt in l['bbox']]
#         ys = [pt.y for pt in l['bbox']]
#         ix1, iy1 = to_px(min(xs), w), to_px(min(ys), h)
#         ix2, iy2 = to_px(max(xs), w), to_px(max(ys), h)
#         print(f"'{l['text']}' -> ({ix1},{iy1})-({ix2},{iy2})")
#         cv2.rectangle(img, (ix1, iy1), (ix2, iy2), (144, 238, 144), thick_inner)

#     # 1️⃣5️⃣ Exibe imagem em janela redimensionável
#     cv2.namedWindow('Debug OCR Cluster', cv2.WINDOW_NORMAL)
#     cv2.imshow('Debug OCR Cluster', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print('Uso: python debug_image.py <caminho_imagem>')
#     else:
#         process_image_with_debug(sys.argv[1])
import sys
import cv2
from services.ocr_cluster import analyze_and_cluster

# Utilitário para converter coordenadas normalizadas (0-1) ou absolutas em pixels
def to_px(val, dim):
    if 0 <= val <= 1:
        return int(val * dim)
    return int(val)


def debug_image(image_path: str):
    """
    Executa a análise e clusterização de OCR na imagem, desenha as bounding boxes
    em uma janela redimensionável e imprime coordenadas no terminal.
    """
    # 1️⃣ Chama o módulo principal para extrair dados
    result = analyze_and_cluster(image_path)

    # 2️⃣ Carrega e redimensiona imagem para a mesma largura usada no módulo
    TARGET_W = 1000
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: não foi possível carregar '{image_path}'")
        return
    ratio = TARGET_W / img.shape[1]
    img = cv2.resize(img, (TARGET_W, int(img.shape[0] * ratio)))
    h, w = img.shape[:2]
    print(f"Imagem carregada e redimensionada para {w}x{h}")

    # 3️⃣ Define espessuras de linha proporcionais ao tamanho da imagem
    base = min(w, h)
    thick_main = max(2, int(base * 0.01))    # 1% do menor lado
    thick_inner = max(1, int(base * 0.005))  # 0.5% do menor lado

    # 4️⃣ Desenha bounding boxes internas (cluster_lines) em verde claro
    for line in result['cluster_lines']:
        xs = [pt.x for pt in line['bbox']]
        ys = [pt.y for pt in line['bbox']]
        x1, y1 = to_px(min(xs), w), to_px(min(ys), h)
        x2, y2 = to_px(max(xs), w), to_px(max(ys), h)
        print(f"Linha: '{line['text']}', bbox pixels: ({x1},{y1})->({x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (144, 238, 144), thick_inner)

    # 5️⃣ Desenha bounding box principal (bbox) em verde forte
    bbox = result.get('bbox')
    if bbox:
        x1, y1 = to_px(bbox['xmin'], w), to_px(bbox['ymin'], h)
        x2, y2 = to_px(bbox['xmax'], w), to_px(bbox['ymax'], h)
        print(f"Cluster bbox principal pixels: ({x1},{y1})->({x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick_main)

    # 6️⃣ Exibe resultado em uma janela redimensionável
    cv2.namedWindow('Debug OCR Cluster', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug OCR Cluster', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Uso: python debug_image.py <caminho_da_imagem>')
    else:
        debug_image(sys.argv[1])
