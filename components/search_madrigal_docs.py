# =========================
# Tool 1: search_madrigal_docs
# =========================
from typing import List
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
import chromadb


class SearchMadrigalDocs(Component):
    display_name = "Search Madrigal Docs"
    description = "Tool: searches the Madrigal documentation Chroma collection (docs KB)."
    icon = "search"
    name = "search_madrigal_docs"
    trace_type = "tool"

    PERSIST_DIR = "./madrigal_web_docs"      # folder where your Chroma DB persists
    COLLECTION = "madrigal_web_docs"        # collection name inside that persist dir

    # ---- Anti-loop / stability settings ----
    TOP_K = 3
    MAX_DOC_CHARS = 1400
    MAX_TOTAL_CHARS = 5200

    inputs = [
        MessageTextInput(name="query", display_name="Query", tool_mode=True),
    ]

    outputs = [
        Output(display_name="Results", name="results", method="run"),
    ]

    def run(self) -> Message:
        query = (self.query or "").strip()
        if not query:
            return Message(text="NO_RESULTS")

        client = chromadb.PersistentClient(path=self.PERSIST_DIR)
        col = client.get_collection(self.COLLECTION)

        res = col.query(query_texts=[query], n_results=self.TOP_K)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        blocks: List[str] = []
        total = 0

        for i, doc in enumerate(docs):
            if not doc:
                continue

            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            src = meta.get("url") or meta.get("source") or meta.get("pdf") or ""

            snippet = doc[: self.MAX_DOC_CHARS].strip()
            header = f"[{i+1}] {src}".strip() if src else f"[{i+1}]"

            piece = f"{header}\n{snippet}"
            if total + len(piece) > self.MAX_TOTAL_CHARS:
                break

            blocks.append(piece)
            total += len(piece)

        if not blocks:
            return Message(text="NO_RESULTS")

        return Message(text="\n\n".join(blocks))
