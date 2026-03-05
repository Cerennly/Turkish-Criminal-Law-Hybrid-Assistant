"""
Generation: Ollama LLM with strict system prompt and deterministic settings.
Temperature=0.1, top_p=0.9 to minimize hallucination.
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TOP_P

SYSTEM_PROMPT = """You are a Turkish Criminal Law expert.
Answer strictly based on provided context.
If the answer is not found in context, respond:
'Bu bilgi yüklenen belgelerde yer almamaktadır.'
Do not speculate.
Do not provide external legal interpretation."""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer (based only on the context above):"""


def get_llm():
    """Ollama LLM with low temperature for deterministic, grounded output (top_p set in config)."""
    return Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=1024,
    )


def build_chain():
    """Build a simple chain: prompt -> llm -> string output."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
    llm = get_llm()
    return prompt | llm | StrOutputParser()
