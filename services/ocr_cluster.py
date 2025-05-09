import os
import re
from dotenv import load_dotenv
import numpy as np
import yaml
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from utils.load_blacklist import load_blacklist

# Importa o leitor de códigos de barra/QR
from services.barcode_reader import ler_codigos_barra_e_qr

import logging
logger = logging.getLogger(__name__)
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

def calculate_overlap_ratio(line_box, cluster_box):
    # ... (mantém igual)
    xA = max(line_box['xmin'], cluster_box['xmin'])
    yA = max(line_box['ymin'], cluster_box['ymin'])
    xB = min(line_box['xmax'], cluster_box['xmax'])
    yB = min(line_box['ymax'], cluster_box['ymax'])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    line_area = (line_box['xmax'] - line_box['xmin']) * (line_box['ymax'] - line_box['ymin'])
    return inter_area / float(line_area) if line_area else 0.0

def bbox_from_line(line):
    xs = [pt.x for pt in line['bbox']]
    ys = [pt.y for pt in line['bbox']]
    return {'xmin': min(xs), 'ymin': min(ys), 'xmax': max(xs), 'ymax': max(ys)}


def read_image_extract_text(image_path):
    """
    Lê a imagem e extrai o texto usando OCR.
    """
    with open(image_path, 'rb') as f:
        ocr_res = vision_client.analyze(
            image_data=f,
            visual_features=[VisualFeatures.READ],
            language='pt'
        )
    lines = []
    if ocr_res.read and ocr_res.read.blocks:
        for block in ocr_res.read.blocks:
            for ln in block.lines:
                txt = ln.text.strip()
                bbox = ln.bounding_polygon
                xs = [pt.x for pt in bbox]
                ys = [pt.y for pt in bbox]
                lines.append({'text': txt, 'bbox': bbox, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys))})
    if lines:
        # Ordena por coordenada Y (de cima para baixo)
        lines.sort(key=lambda l: l['cy'])   
        
    return lines

def find_recipient_anchor(lines):
    """
    Encontra a âncora do destinatário na lista de linhas OCR.
    """
    recipient_pattern = re.compile(r"\b(destinat[áa]rio|recipient)\b", re.IGNORECASE)
    for i, l in enumerate(lines):
        if recipient_pattern.search(l['text']):
            txt = l['text']
            logger.debug(f"Âncora Destinatário encontrada: '{txt}'")
            return i
    return None

def find_cep_anchor(lines, cep_da_barcode = None):
    """
    Encontra âncoras de CEP na lista de linhas OCR.
    """
    # regex para qualquer CEP
    cep_pattern = re.compile(r"\b\d{5}-?\d{3}\b")
    ceps = []
    if cep_da_barcode:
        for i, l in enumerate(lines):
            if cep_da_barcode in l['text']:
                ceps.append({'idx': i, 'cep': cep_da_barcode, 'cy': l['cy']})
        if not ceps:
            logger.debug("CEP da barcode não encontrado via OCR. Fazendo fallback regex.")
    if not cep_da_barcode or not ceps:
        # regex para qualquer CEP
        for i, l in enumerate(lines):
            m = cep_pattern.search(l['text'])
            if m:
                ceps.append({'idx': i, 'cep': m.group(), 'cy': l['cy']})
    if not ceps:
         logger.debug("Nenhum CEP encontrado.")
    else:
         logger.debug(f"CEPs encontrados: {[c['cep'] for c in ceps]}")
    return ceps

def find_NER_recipient(lines):
    """
    Encontra todos os possíveis destinatários usando NER na lista de linhas OCR.
    """
    # load blacklist de destinatários
    blacklist = load_blacklist()

    full_text = "\n".join([l['text'] for l in lines])
    ner = text_client.recognize_entities([full_text], language='pt')[0]
    entities = [{'text': ent.text, 'category': ent.category} for ent in ner.entities]

    candidates = []
    for ent in ner.entities:
        if ent.category == 'Person':
            # ignora se também Organization
            if any(o['text'] == ent.text and o['category'] == 'Organization' for o in entities) or ent.text in blacklist:
                continue
            # localiza na linha OCR
            for i, l in enumerate(lines):
                if ent.text in l['text']:
                    logger.debug(f"Destinatário (NER) candidato encontrado: '{l['text']}'")
                    candidates.append({'index': i, 'text': l['text'], 'bbox': l['bbox'], 'cy': l['cy']})
    if not candidates:
         logger.debug("Nenhum destinatário detectado via NER.")
    return candidates
    
def find_achors(lines, cep_da_barcode = None):
    """
    Encontra âncoras na lista de linhas OCR.
    """

    # Encontra âncora do destinatário
    recipient_anchor_idx = find_recipient_anchor(lines)
    # Encontra Ancora do CEP
    ceps = find_cep_anchor(lines, cep_da_barcode)
    # Encontra possíveis destinatários via NER
    person_candidates = find_NER_recipient(lines)
    cep = None
    if recipient_anchor_idx is not None:
        # Encontrar o CEP mais próximo da âncora do destinatário
        below_ceps = [c for c in ceps if c['cy'] > lines[recipient_anchor_idx]['cy']]
        if below_ceps:
            closest_cep = min(below_ceps, key=lambda c: c['cy'] - lines[recipient_anchor_idx]['cy'])
        else:
            closest_cep = ceps[0] if ceps else None

        if closest_cep:
            cep_idx = closest_cep['idx']
            # Considerar somente linhas entre a âncora e o CEP
            spatial_lines = [l for l in lines if lines[recipient_anchor_idx]['cy'] <= l['cy'] <= lines[cep_idx]['cy']]
            bbox = {
                'xmin': min(l['bbox'][0].x for l in spatial_lines),
                'ymin': min(l['bbox'][0].y for l in spatial_lines),
                'xmax': max(l['bbox'][2].x for l in spatial_lines),
                'ymax': max(l['bbox'][2].y for l in spatial_lines),
            }
            recipient_candidates = [l for l in person_candidates if l['cy'] < lines[cep_idx]['cy'] and l['cy'] > lines[recipient_anchor_idx]['cy']]
            recipient = recipient_candidates[0] if recipient_candidates else lines[recipient_anchor_idx]
            
            return {'bbox': bbox, 'cep':lines[cep_idx], 'recipient': recipient}
    else:
        # Usar o primeiro candidato de Person
        if person_candidates:
            recipient = person_candidates[0]
            recipient_idx = recipient['index']

            # Encontrar o CEP mais próximo do candidato
            below_ceps = [c for c in ceps if c['cy'] > lines[recipient_idx]['cy']]
            if below_ceps:
                closest_cep = min(below_ceps, key=lambda c: c['cy'] - lines[recipient_idx]['cy'])
            else:
                closest_cep = ceps[0] if ceps else None

            if closest_cep:
                cep_idx = closest_cep['idx']
                # Considerar somente linhas entre o candidato e o CEP
                spatial_lines = [l for l in lines if lines[recipient_idx]['cy'] <= l['cy'] <= lines[cep_idx]['cy']]
                bbox = {
                    'xmin': min(l['bbox'][0].x for l in spatial_lines),
                    'ymin': min(l['bbox'][0].y for l in spatial_lines),
                    'xmax': max(l['bbox'][2].x for l in spatial_lines),
                    'ymax': max(l['bbox'][2].y for l in spatial_lines),
                }
                return {'bbox': bbox, 'cep':lines[cep_idx], 'recipient': recipient}

    return {'error': 'Nenhuma âncora ou destinatário encontrado.'}
    

def analyze_and_cluster(image_path, iou_threshold=0.05):
    """
    1) Usa o leitor de códigos de barra/QR para extrair CEP antes de qualquer OCR.
    2) Procura pela palavra âncora 'destinatário' ou 'recipient' para identificar o nome.
    3) Se não encontrar, segue buscando Person via NER.
    """
    #blacklist de destinatários
    blacklist = load_blacklist()
    
    # ---- Etapa 1: detectar CEP via barcode_reader ----
    barcode = ler_codigos_barra_e_qr(image_path)
    cep_da_barcode = barcode.get("cep") if barcode else None
    if cep_da_barcode:
        logger.debug(f"CEP detectado via código de barra: {cep_da_barcode}")
    else:
        logger.debug("Nenhum CEP via código de barra. Usando OCR+regex.")

    # ---- Etapa 1: ler imagem e extrair texto ----
    lines = read_image_extract_text(image_path)
    if not lines:
        return {'error': 'Nenhuma linha OCR detectada.'}
    
    anchors = find_achors(lines, cep_da_barcode)
    cep = anchors.get('cep')
    recipient = anchors.get('recipient')
    main_bbox = anchors.get('bbox')
    if not main_bbox:
        return {'error': 'Nenhuma âncora ou destinatário encontrado.'}
    
    # Identifica linhas OCR dentro da main_bbox usando overlap ratio
    cluster_idxs = set()
    for i, line in enumerate(lines):
        overlap = calculate_overlap_ratio(bbox_from_line(line), main_bbox)
        if overlap >= 0.3:
            cluster_idxs.add(i)
    
    # Atualiza main_bbox com base nas linhas identificadas
    for idx in cluster_idxs:
        bb = bbox_from_line(lines[idx])
        main_bbox = {
            'xmin': min(main_bbox['xmin'], bb['xmin']),
            'ymin': min(main_bbox['ymin'], bb['ymin']),
            'xmax': max(main_bbox['xmax'], bb['xmax']),
            'ymax': max(main_bbox['ymax'], bb['ymax'])
        }

    
    return {
        'recipient': recipient,
        'cep': cep,
        'bbox': main_bbox,
        'cluster_lines': [lines[i] for i in sorted(cluster_idxs)],

    }
