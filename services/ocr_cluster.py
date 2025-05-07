import os
import re
from dotenv import load_dotenv
import numpy as np
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Carrega variáveis de ambiente (Azure)
load_dotenv()
API_KEY = os.getenv("MULTISERVICE_API_KEY")
ENDPOINT = os.getenv("MULTISERVICE_ENDPOINT")

if not API_KEY or not ENDPOINT:
    raise EnvironmentError("Defina MULTISERVICE_API_KEY e MULTISERVICE_ENDPOINT no .env")

# Inicializa clientes Azure
vision_client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(API_KEY)
)
text_client = TextAnalyticsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(API_KEY)
)


def calculate_iou(boxA, boxB):
    """
    Calcula Intersection over Union (IoU) entre duas bounding boxes.
    Cada box é um dict com xmin, ymin, xmax, ymax.
    """
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])

    # Compute area of intersection
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0  # no overlap

    # Compute areas of each box
    boxA_area = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
    boxB_area = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])

    # Compute IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def bbox_from_line(line):
    """Converte pontos da bbox para dict (xmin, ymin, xmax, ymax)."""
    xs = [pt.x for pt in line['bbox']]
    ys = [pt.y for pt in line['bbox']]
    return {
        'xmin': min(xs),
        'ymin': min(ys),
        'xmax': max(xs),
        'ymax': max(ys)
    }


def analyze_and_cluster(image_path, iou_threshold=0.05):
    """
    Executa OCR + NER, identifica destinatário (Person não confuso com Organization)
    e o CEP mais próximo abaixo dele. Usa IoU para adicionar linhas próximas ao cluster.
    Retorna dados para debug_image desenhar.
    """

    # 1️⃣ OCR - Analisar imagem com Azure Vision
    with open(image_path, 'rb') as f:
        ocr_res = vision_client.analyze(
            image_data=f,
            visual_features=[VisualFeatures.READ],
            language='pt'
        )

    # 2️⃣ Extrair todas as linhas OCR em lista ordenada por Y (topo → base)
    lines = []
    if ocr_res.read and ocr_res.read.blocks:
        for block in ocr_res.read.blocks:
            for ln in block.lines:
                txt = ln.text.strip()
                bbox = ln.bounding_polygon  # lista de 4 pontos normalizados
                xs = [pt.x for pt in bbox]
                ys = [pt.y for pt in bbox]
                lines.append({
                    'text': txt,
                    'bbox': bbox,
                    'cx': float(np.mean(xs)),
                    'cy': float(np.mean(ys))
                })

    if not lines:
        print("❌ Nenhuma linha OCR detectada.")
        return {'error': 'Nenhuma linha OCR detectada.'}

    # ✅ Ordena por posição Y (de cima para baixo)
    lines.sort(key=lambda l: l['cy'])

    # 3️⃣ Rodar NER para localizar entidades
    full_text = "\n".join([l['text'] for l in lines])
    ner_result = text_client.recognize_entities([full_text], language='pt')[0]

    entities = [{'text': ent.text, 'category': ent.category} for ent in ner_result.entities]

    dest_idx = None
    for ent in ner_result.entities:
        if ent.category == 'Person':
            is_confused = any(
                other_ent for other_ent in entities
                if other_ent['text'] == ent.text and other_ent['category'] == 'Organization'
            )
            if is_confused:
                print(f"⚠️ Ignorando '{ent.text}' porque também foi identificado como Organization.")
                continue
            for i, l in enumerate(lines):
                if ent.text in l['text']:
                    dest_idx = i
                    break
            if dest_idx is not None:
                break

    if dest_idx is not None:
        print(f"✅ Destinatário encontrado: '{lines[dest_idx]['text']}' na linha {dest_idx}")
    else:
        print("⚠️ Nenhum destinatário (Person válido) detectado.")

    # 4️⃣ Procurar todos CEPs via regex (xxxxx-xxx ou xxxxxxxx)
    ceps = []
    cep_pattern = re.compile(r"\b\d{5}-?\d{3}\b")
    for i, l in enumerate(lines):
        match = cep_pattern.search(l['text'])
        if match:
            ceps.append({
                'idx': i,
                'cep': match.group(),
                'cy': l['cy']
            })

    if not ceps:
        print("⚠️ Nenhum CEP encontrado.")
    else:
        print(f"🔢 {len(ceps)} CEP(s) encontrado(s): {[c['cep'] for c in ceps]}")

    # 5️⃣ Encontrar o CEP mais próximo e abaixo do destinatário (se existir)
    cep_idx = None
    if dest_idx is not None and ceps:
        ceps_below = [c for c in ceps if c['cy'] > lines[dest_idx]['cy']]
        if ceps_below:
            closest_cep = min(ceps_below, key=lambda c: c['cy'] - lines[dest_idx]['cy'])
            cep_idx = closest_cep['idx']
            print(f"📍 CEP mais próximo abaixo: '{lines[cep_idx]['text']}' na linha {cep_idx}")
        else:
            print("⚠️ Nenhum CEP abaixo do destinatário. Usando o primeiro CEP encontrado.")
            cep_idx = ceps[0]['idx']
    elif ceps:
        cep_idx = ceps[0]['idx']
        print(f"📍 Nenhum destinatário, usando o primeiro CEP '{lines[cep_idx]['text']}'.")

    # 6️⃣ Inicializa o cluster principal com destinatário + cep
    cluster_idxs = set()
    if dest_idx is not None:
        cluster_idxs.add(dest_idx)
    if cep_idx is not None:
        cluster_idxs.add(cep_idx)

    if not cluster_idxs:
        print("⚠️ Nada para clusterizar (nenhuma âncora encontrada).")
        return {'error': 'Nada encontrado para clusterizar.'}

    # 7️⃣ Inicializa bbox global com as âncoras
    initial_lines = [lines[idx] for idx in cluster_idxs]
    xs = [pt.x for l in initial_lines for pt in l['bbox']]
    ys = [pt.y for l in initial_lines for pt in l['bbox']]
    bbox_global = {
        'xmin': min(xs),
        'ymin': min(ys),
        'xmax': max(xs),
        'ymax': max(ys)
    }

    # 8️⃣ Rodar expansão por IoU para adicionar linhas próximas
    for i, line in enumerate(lines):
        if i in cluster_idxs:
            continue  # já está no cluster
        line_bbox = bbox_from_line(line)
        iou = calculate_iou(line_bbox, bbox_global)
        if iou >= iou_threshold:
            print(f"➕ Linha '{line['text']}' adicionada ao cluster (IoU={iou:.3f})")
            cluster_idxs.add(i)
            # Atualiza a bbox global para expandir o cluster
            bbox_global = {
                'xmin': min(bbox_global['xmin'], line_bbox['xmin']),
                'ymin': min(bbox_global['ymin'], line_bbox['ymin']),
                'xmax': max(bbox_global['xmax'], line_bbox['xmax']),
                'ymax': max(bbox_global['ymax'], line_bbox['ymax'])
            }

    # 9️⃣ Retorna resultado
    cluster_lines = [lines[idx] for idx in sorted(cluster_idxs)]
    return {
        'cluster_lines': cluster_lines,
        'bbox': bbox_global
    }
