import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
from langchain.document_loaders import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document

# Tracing (optional)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

local_llm = "phi3"

# INDEX
file_paths = [
    "C:\\Users\\ayush\\OneDrive\\Desktop\\llms_mod\\Attention (1).pdf",
    "C:\\Users\\ayush\\OneDrive\\Desktop\\llms_mod\\AYUSH_DISSERTATION_Final.pdf",
    "C:\\Users\\ayush\\OneDrive\\Desktop\\llms_mod\\AYUSH_Nath_TIWARI_RESUME .pdf",
]

docs = [PyPDFLoader(file_path).load() for file_path in file_paths]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# MODEL
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt for retrieval grader
retrieval_prompt = PromptTemplate(
    template="""system You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     user
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n assistant
    """,
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

# Prompt for answer generation
generation_prompt = PromptTemplate(
    template="""system You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise user
    Question: {question}
    Context: {context}
    Answer: assistant""",
    input_variables=["question", "document"],
)

rag_chain = generation_prompt | llm | StrOutputParser()

# Hallucination Grader
hallucination_prompt = PromptTemplate(
    template=""" system You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation. user
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  assistant""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_prompt | llm | JsonOutputParser()

# Answer Grader
answer_prompt = PromptTemplate(
    template="""system You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     user Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} assistant""",
    input_variables=["generation", "question"],
)

answer_grader = answer_prompt | llm | JsonOutputParser()

# Router
router_prompt = PromptTemplate(
    template="""system You are an expert at routing a
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM
    agents, prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
    no premable or explaination. Question to route: {question} assistant""",
    input_variables=["question"],
)

question_router = router_prompt | llm | JsonOutputParser()

# Web Search
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

# LANGCGARPH
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

def retrieve(state):
    # Retrieve documents from vectorstore
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    # Generate answer using RAG on retrieved documents
    question = state["question"]
    documents = state.get("documents", [])
    
    if documents:
        generation = rag_chain.invoke({"context": documents, "question": question})
        # Remove '\\', '\n', and any other unwanted characters
        generation = generation.replace('\\', '').replace('\n', '').strip()
    else:
        generation = ""  # Set default value if documents is empty
        
    return {"generation": generation, "documents": documents, "question": question}






def grade_documents(state):
    # Determine relevance of retrieved documents to the question
    question = state["question"]
    documents = state.get("documents", [])  # Use get() method to handle missing key
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score["score"]
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    if filtered_docs:  # Check if filtered_docs is not empty
        generation = rag_chain.invoke({"context": filtered_docs, "question": question})
    else:
        generation = ""  # Set default value if filtered_docs is empty
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "generation": generation}


def web_search(state):
    # Perform web search based on the question
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

# Conditional edge decisions
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RETRIEVE---")
        return "retrieve_documents"  # Update to match the node name in the workflow


def decide_to_generate(state):
    question = state["question"]
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"

def grade_generation_v_documents_and_question(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]
    if grade == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes and edges
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Define conditional entry point
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "retrieve": "retrieve",  # Update to match the node name in the workflow
    },
)

# Define edges based on conditions
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)
# Compile the workflow
app = workflow.compile()

# Print nodes in the workflow
print("Nodes in the workflow:", workflow.nodes)

# Test the workflow
inputs = {"question": "what is ECNN",}
for output in app.stream(inputs):
    print(f"Finished running: {output}")
    print(output["generation"])
