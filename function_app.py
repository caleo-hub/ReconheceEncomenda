import json
import logging
import os
import re
import tempfile

import numpy as np
from sklearn.cluster import DBSCAN
import azure.functions as func
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# 📦 Carregar variáveis de ambiente
load_dotenv()

MULTISERVICE_API_KEY = os.getenv("MULTISERVICE_API_KEY")
MULTISERVICE_ENDPOINT = os.getenv("MULTISERVICE_ENDPOINT")
if not MULTISERVICE_API_KEY or not MULTISERVICE_ENDPOINT:
    raise EnvironmentError("Variáveis MULTISERVICE_API_KEY e MULTISERVICE_ENDPOINT não estão definidas.")

vision_client = ImageAnalysisClient(
    endpoint=MULTISERVICE_ENDPOINT,
    credential=AzureKeyCredential(MULTISERVICE_API_KEY)
)
text_client = TextAnalyticsClient(
    endpoint=MULTISERVICE_ENDPOINT,
    credential=AzureKeyCredential(MULTISERVICE_API_KEY)
)

app = func.FunctionApp()

BLACKLIST = {"shein", "magalu", "mercado livre", "amazon"}

@app.route(route="read_ticket_package", auth_level=func.AuthLevel.ANONYMOUS)
def read_ticket_package(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('📦 Processando imagem recebida para leitura de pacote.')
    try:
        file = req.files.get('file')
        if not file:
            return func.HttpResponse("Nenhum arquivo enviado. Envie a imagem no campo 'file'.", status_code=400)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file)
            temp_path = temp_file.name

        with open(temp_path, "rb") as image_stream:
            result = vision_client.analyze(
                image_data=image_stream,
                visual_features=[VisualFeatures.READ],
                language="pt"
            )

        linhas_ocr = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    text = line.text.strip()
                    bbox = line.bounding_polygon
                    ys = [pt.y for pt in bbox]
                    xs = [pt.x for pt in bbox]
                    linhas_ocr.append({
                        "text": text,
                        "bbox": bbox,
                        "top": min(ys),
                        "center_x": sum(xs) / len(xs),
                        "center_y": sum(ys) / len(ys)
                    })
        linhas_ocr.sort(key=lambda x: x["top"])

        full_text = "\n".join([l["text"] for l in linhas_ocr])
        response = text_client.recognize_entities([full_text], language="pt")[0]

        destinatario = None
        dest_idx = None
        for entity in response.entities:
            if entity.category == "Person":
                texto_entity = entity.text.lower()
                is_organization = any(org_ent.text.lower() == texto_entity for org_ent in response.entities if org_ent.category == "Organization")
                if texto_entity not in BLACKLIST and not is_organization:
                    destinatario = entity.text
                    for i, l in enumerate(linhas_ocr):
                        if destinatario in l["text"]:
                            dest_idx = i
                            break
                    break

        cep_matches = []
        for i, l in enumerate(linhas_ocr):
            m = re.search(r"\b\d{5}-?\d{3}\b", l["text"])
            if m:
                cep_matches.append((i, m.group()))
        cep_encontrado = None
        cep_idx = None
        if cep_matches:
            if dest_idx is not None:
                cep_idx, cep_encontrado = min(cep_matches, key=lambda x: abs(x[0] - dest_idx))
            else:
                cep_idx, cep_encontrado = cep_matches[0]

        coords = [[l["center_x"], l["center_y"]] for l in linhas_ocr]
        clustering = DBSCAN(eps=50, min_samples=1).fit(coords)
        for i, l in enumerate(linhas_ocr):
            l["cluster"] = clustering.labels_[i]

        cluster_dest = linhas_ocr[dest_idx]["cluster"] if dest_idx is not None else None
        cluster_cep = linhas_ocr[cep_idx]["cluster"] if cep_idx is not None else None
        if cluster_dest is not None and cluster_cep is not None:
            relevant = {cluster_dest, cluster_cep} if cluster_dest != cluster_cep else {cluster_dest}
        elif cluster_cep is not None:
            relevant = {cluster_cep}
        else:
            relevant = {cluster_dest} if cluster_dest is not None else set()

        linhas_cluster = [l for l in linhas_ocr if l.get("cluster") in relevant]
        if linhas_cluster:
            all_x = [pt.x for l in linhas_cluster for pt in l['bbox']]
            all_y = [pt.y for l in linhas_cluster for pt in l['bbox']]
            cluster_bbox = {
                'xmin': min(all_x), 'ymin': min(all_y),
                'xmax': max(all_x), 'ymax': max(all_y)
            }
            logging.info(f"📦 Cluster relevante bbox: {cluster_bbox}")

        ocr_texto_final = "\n".join([l["text"] for l in linhas_cluster])

        resultado = {
            "destinatario": destinatario or "",
            "cep": cep_encontrado or "",
            "ocr": ocr_texto_final
        }
        return func.HttpResponse(
            json.dumps(resultado, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"❌ Erro ao processar: {e}")
        return func.HttpResponse(f"Erro interno: {str(e)}", status_code=500)
