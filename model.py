from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =========================
# Carregar documentos
# =========================
loader1 = PyPDFLoader("cf.pdf")
data1 = loader1.load()
for doc in data1:
    doc.metadata["fonte"] = "Constituição Federal"

loader2 = PyPDFLoader("clt_e_normas_correlatas_1ed.pdf")
data2 = loader2.load()
for doc in data2:
    doc.metadata["fonte"] = "CLT"

documentos = data1 + data2

# =========================
# Splitter
# =========================
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "Art.", "§", "Inciso", "Alínea", ".", "\n"],
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documentos)

# =========================
# Vetorização
# =========================
embedding_model = OpenAIEmbeddings(
    api_key="<YOUR_KEY>",
    model="text-embedding-3-small",
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# =========================
# LLM + Prompt
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=300,
    api_key="YOUR_KEY"
)

template = """
Você é um assistente que responde perguntas sobre a Constituição Federal e a CLT.
Sua missão é responder à pergunta abaixo usando SOMENTE o texto fornecido nos documentos,
ignorando qualquer sumário, nota de rodapé ou comentário.
Se a resposta não estiver clara nos documentos, diga que não sabe.
Sua resposta deve ser simples, didática e divertida, usando emojis.

---
Documentos:
{context}
---

Pergunta: {question}

Resposta:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)