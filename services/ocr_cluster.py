import os
import re
import cv2
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Carrega variáveis de ambiente para Azure
load_dotenv()
API_KEY = os.getenv("MULTISERVICE_API_KEY")
ENDPOINT = os.getenv("MULTISERVICE_ENDPOINT")
if not API_KEY or not ENDPOINT:
    raise EnvironmentError("Defina MULTISERVICE_API_KEY e MULTISERVICE_ENDPOINT no .env")

# Inicializa clientes Azure Vision e Text Analytics
vision_client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(API_KEY)
)
text_client = TextAnalyticsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(API_KEY)
)

# Blacklist para evitar falsos positivos (empresas confundidas como Person)
BLACKLIST = {"shein", "magalu", "mercado livre", "amazon"}


def analyze_and_cluster(
    image_path: str,
    resize_width: int = 1000,
    eps: float = 0.01,
    min_samples: int = 1
) -> dict:
    """
    Processa a imagem: OCR, NER e clusterização espacial em torno do destinatário e CEP.
    Retorna:
      destinatario: str ou None
      cep: str ou None
      lines: lista de dicts{text, bbox(normalizado), cx, cy, cluster}
      cluster_lines: subset de lines no(s) cluster(s) relevante(s)
      bbox: dict{xmin, ymin, xmax, ymax} normalizado
    """
    # 1️⃣ Carrega e redimensiona mantendo proporção
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    h0, w0 = img.shape[:2]
    ratio = resize_width / float(w0)
    img_resized = cv2.resize(img, (resize_width, int(h0 * ratio)))

    # 2️⃣ Salva temporária para envio ao OCR
    tmp = image_path + ".tmp.jpg"
    cv2.imwrite(tmp, img_resized)
    
    # 3️⃣ OCR com Azure Vision
    with open(tmp, 'rb') as f:
        ocr_res = vision_client.analyze(
            image_data=f,
            visual_features=[VisualFeatures.READ],
            language='pt'
        )
    os.remove(tmp)

    # 4️⃣ Extrai linhas OCR: texto, bbox normalizado e centroides
    lines = []
    if ocr_res.read and ocr_res.read.blocks:
        for block in ocr_res.read.blocks:
            for ln in block.lines:
                txt = ln.text.strip()
                bbox = ln.bounding_polygon
                xs = [pt.x for pt in bbox]
                ys = [pt.y for pt in bbox]
                lines.append({
                    'text': txt,
                    'bbox': bbox,
                    'cx': float(np.mean(xs)),
                    'cy': float(np.mean(ys))
                })
    if not lines:
        return { 'destinatario': None, 'cep': None, 'lines': [], 'cluster_lines': [], 'bbox': None }

    # 5️⃣ NER para identificar destinatário (Person não Organization/blacklist)
    full_text = '\n'.join([l['text'] for l in lines])
    ner = text_client.recognize_entities([full_text], language='pt')[0]
    dest_idx = None
    for ent in ner.entities:
        if ent.category == 'Person':
            ent_low = ent.text.lower()
            is_org = any(e.text.lower() == ent_low and e.category == 'Organization' for e in ner.entities)
            if ent_low not in BLACKLIST and not is_org:
                for i, l in enumerate(lines):
                    if ent.text in l['text']:
                        dest_idx = i
                        break
                if dest_idx is not None:
                    break

    # 6️⃣ Regex para localizar todos os CEPs e escolher o mais próximo
    ceps = [(i, re.search(r"\b\d{5}-?\d{3}\b", l['text'])) for i, l in enumerate(lines)]
    ceps = [(i, m.group()) for i, m in ceps if m]
    cep_idx = None
    cep_val = None
    if ceps:
        if dest_idx is not None:
            cep_idx, cep_val = min(ceps, key=lambda x: abs(x[0] - dest_idx))
        else:
            cep_idx, cep_val = ceps[0]

    # 7️⃣ Clusterização espacial via DBSCAN nos centroides normalizados
    coords = np.array([[l['cx'], l['cy']] for l in lines])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    for i, l in enumerate(lines):
        l['cluster'] = int(labels[i])

    # 8️⃣ Determina clusters relevantes (destinatário e CEP)
    relevant = set()
    if dest_idx is not None:
        relevant.add(lines[dest_idx]['cluster'])
    if cep_idx is not None:
        relevant.add(lines[cep_idx]['cluster'])

    # 9️⃣ Filtra linhas dos clusters relevantes (fallback: todas se vazio)
    cluster_lines = [l for l in lines if l['cluster'] in relevant]
    if not cluster_lines:
        cluster_lines = lines

    # 🔟 Calcula bbox normalizado do cluster relevante
    xs = [pt.x for l in cluster_lines for pt in l['bbox']]
    ys = [pt.y for l in cluster_lines for pt in l['bbox']]
    bbox = {
        'xmin': float(min(xs)),
        'ymin': float(min(ys)),
        'xmax': float(max(xs)),
        'ymax': float(max(ys))
    } if xs and ys else None

    return {
        'destinatario': lines[dest_idx]['text'] if dest_idx is not None else None,
        'cep': cep_val,
        'lines': lines,
        'cluster_lines': cluster_lines,
        'bbox': bbox
    }