import json
import logging
import os
import tempfile

import azure.functions as func
from dotenv import load_dotenv

from services.ocr_cluster import analyze_and_cluster  

# configura o logger da função (opcional)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = func.FunctionApp()

@app.route(route="read_ticket_package", auth_level=func.AuthLevel.ANONYMOUS)
def read_ticket_package(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processando imagem recebida para leitura de pacote.')
    try:
        file = req.files.get('file')
        if not file:
            return func.HttpResponse(
                "Nenhum arquivo enviado. Envie a imagem no campo 'file'.",
                status_code=400
            )

        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file)
            temp_path = temp_file.name

        # Chama o OCR/Cluster
        result = analyze_and_cluster(temp_path)

        # Se houve erro na extração
        if 'error' in result:
            return func.HttpResponse(
                result['error'],
                status_code=400,
                mimetype="text/plain"
            )

        # Prepara texto OCR do cluster
        ocr_cluster_text = "\n".join([line["text"] for line in result.get('cluster_lines', [])])

        # Monta resposta completa
        response_data = {
            # destinatário pode vir como dict {'text':..., 'bbox':..., ...}
            "recipient": (result['recipient']['text']
                          if isinstance(result.get('recipient'), dict)
                          else result.get('recipient')),
            # CEP extraído
            "cep": (result['cep']['text']
                    if isinstance(result.get('cep'), dict)
                    else result.get('cep')),
            # todo o texto do cluster
            "ocr_cluster": ocr_cluster_text,
            # bbox principal
            "bbox": result.get('bbox'),
            # linhas individuais do cluster (texto + bbox de cada linha)
            "cluster_lines": [
                {
                    "text": line["text"],
                    "bbox": [
                        {"x": pt.x, "y": pt.y}
                        for pt in line["bbox"]
                    ]
                }
                for line in result.get('cluster_lines', [])
            ]
        }

        return func.HttpResponse(
            body=json.dumps(response_data, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Erro ao processar: {e}", exc_info=True)
        return func.HttpResponse(
            f"Erro interno: {str(e)}",
            status_code=500,
            mimetype="text/plain"
        )
