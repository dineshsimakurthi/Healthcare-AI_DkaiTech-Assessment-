"""
PART 2b - RAG QUESTION ANSWERING with a LangChain LCEL chain

The whole idea in one line:

    question -> retriever (Chroma) -> prompt (context + question) -> LLM -> answer

Run:  python src/rag_chain.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from build_index import CHROMA_DIR, get_embeddings

NOT_FOUND = "I could not find this information in the provided documents."

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful healthcare information assistant.
Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "{not_found}"
Keep the answer short and clear. This is general information, not a diagnosis.

Risk guidance: {risk_guidance}

Context:
{context}

Question: {question}

Answer:"""
).partial(not_found=NOT_FOUND)


def get_llm():
    """Build the configured LLM. Set LLM_PROVIDER in .env to openai, groq, or gemini."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing. Add it to .env.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            api_key=api_key,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is missing. Add it to .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0,
            google_api_key=api_key,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def format_docs(docs):
    """Join the retrieved chunks into one text block for the prompt."""
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(llm=None, embeddings=None):
    # Retriever: finds the 5 most similar chunks in Chroma
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR),
                         embedding_function=embeddings or get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = llm or get_llm()

    # Step A: question -> prompt -> LLM -> text
    answer_chain = PROMPT | llm | StrOutputParser()

    # Step B: run the retriever and keep the question, so we can show the sources later
    chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        .assign(answer=RunnableLambda(
            lambda x: {
                "context": format_docs(x["docs"]),
                "question": x["question"],
                "risk_guidance": (
                    "This is high diabetes risk. Clearly say that the person should contact a doctor immediately or seek urgent medical care, especially for severe symptoms such as severe thirst, frequent urination, vomiting, abdominal pain, confusion, or difficulty breathing. Give only 2-3 brief, safe lifestyle changes."
                    if " high diabetes risk " in x["question"].lower()
                    else "This is moderate diabetes risk. Advise booking a medical consultation promptly and give 2-3 brief lifestyle changes."
                    if " moderate diabetes risk " in x["question"].lower()
                    else "This is low diabetes risk. Recommend routine checkups and a few simple preventive lifestyle changes."
                ),
            }
        ) | answer_chain)
    )
    return chain


def ask(chain, question: str) -> dict:
    """Returns {"answer": str, "sources": [file names]}"""
    out = chain.invoke(question)
    answer = out["answer"].strip()
    if NOT_FOUND in answer:                     # no answer -> no sources to show
        return {"answer": NOT_FOUND, "sources": []}
    sources = sorted({d.metadata["source"] for d in out["docs"]})
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    rag = build_rag_chain()
    print("Ask a healthcare question (type 'exit' to quit)")
    while (q := input("\nQ: ").strip()) and q.lower() != "exit":
        result = ask(rag, q)
        print("A:", result["answer"])
        print("Sources:", ", ".join(result["sources"]) or "none")
