from typing import List
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message

import chromadb
from langchain_openai import OpenAIEmbeddings


def _to_text(x) -> str:
    if isinstance(x, Message):
        return x.text or ""
    return str(x or "")


class SearchMadrigalDocs(Component):
    display_name = "Search Madrigal Docs"
    description = "Tool: searches Madrigal HTML/docs Chroma collection."
    icon = "search"
    name = "search_madrigal_docs"
    trace_type = "tool"

    PERSIST_DIR = "./madrigal_html"
    COLLECTION = "madrigal_html"
    TOP_K = 3

    MAX_DOC_CHARS = 1400
    MAX_TOTAL_CHARS = 5200

    inputs = [
        MessageTextInput(name="query", display_name="Query", tool_mode=True),
        SecretStrInput(name="openai_api_key", display_name="OpenAI API Key", required=True, load_from_db=True),
    ]

    outputs = [
        Output(display_name="Results", name="results", method="run"),
    ]

    def run(self) -> Message:
        try:
            query = _to_text(self.query).strip()
            if not query:
                return Message(text="NO_RESULTS")

            api_key = _to_text(self.openai_api_key).strip()
            if not api_key:
                return Message(text="ERROR(search_madrigal_docs): Missing OpenAI API key")

            emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
            q_emb = emb.embed_query(query)

            client = chromadb.PersistentClient(path=self.PERSIST_DIR)
            col = client.get_collection(self.COLLECTION)

            res = col.query(query_embeddings=[q_emb], n_results=self.TOP_K)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            if not docs:
                return Message(text="NO_RESULTS")

            blocks: List[str] = []
            total = 0

            for i, doc in enumerate(docs):
                if not doc:
                    continue
                meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                src = meta.get("url") or meta.get("source") or meta.get("path") or ""
                snippet = doc[: self.MAX_DOC_CHARS].strip()

                header = f"[{i+1}] {src}".strip() if src else f"[{i+1}]"
                piece = f"{header}\n{snippet}"

                if total + len(piece) > self.MAX_TOTAL_CHARS:
                    break
                blocks.append(piece)
                total += len(piece)

            return Message(text="\n\n".join(blocks) if blocks else "NO_RESULTS")

        except Exception as e:
            return Message(text=f"ERROR(search_madrigal_docs): {type(e).__name__}: {e}")
