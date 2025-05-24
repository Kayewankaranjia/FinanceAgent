
# === Cypher Utilities ===
import json

class CypherUtils:

    @staticmethod
    def create_node(label: str, properties: dict) -> str:
        props = ', '.join([f"{k}: {json.dumps(v)}" for k, v in properties.items()])
        return f"MERGE (:{label} {{{props}}});"

    @staticmethod
    def create_relationship(source_label: str, source_id: str, target_label: str, target_id: str, rel_type: str) -> str:
        return f"MATCH (a:{source_label} {{id: '{source_id}'}}), (b:{target_label} {{id: '{target_id}'}}) MERGE (a)-[:{rel_type}]->(b);"
