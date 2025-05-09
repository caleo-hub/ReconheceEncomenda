import yaml

def load_blacklist(file_path='nao_destinatario.yaml'):
    """
    Carrega a blacklist de nomes que não devem ser considerados destinatários.

    Args:
        file_path (str): Caminho para o arquivo YAML contendo a blacklist.

    Returns:
        set: Um conjunto de nomes em lowercase que fazem parte da blacklist.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nao_dest = yaml.safe_load(f).get('blacklist', {})
        # Une todas as empresas/organizações em um único conjunto em lowercase
        blacklist = set(
            name.lower() for category in nao_dest.values() for name in category
        )
        return blacklist
    except FileNotFoundError:
        print(f"Arquivo {file_path} não encontrado.")
        return set()
    except yaml.YAMLError as e:
        print(f"Erro ao processar o arquivo YAML: {e}")
        return set()