from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Load environment variables from a .env file
load_dotenv()

# Step 1: Load documents from the specified directory
documents = SimpleDirectoryReader("pdf/").load_data()

# Step 2: Split documents into smaller chunks (nodes) for efficient processing
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Step 3: Configure the LLM and embedding models
Settings.llm = OpenAI(model="gpt-3.5-turbo")  # Language model for NLP tasks
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")  # Embedding model for vector representation

# Step 4: Build indices for summarization and vector-based retrieval
summary_index = SummaryIndex(nodes)  # Creates a summarization index
vector_index = VectorStoreIndex(nodes)  # Creates a vector-based index for semantic search

# Step 5: Create query engines from the indices
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",  # Use tree summarization mode
    use_async=True,  # Enable faster query generation by leveraging asynchronous processing
)

vector_query_engine = vector_index.as_query_engine()  # Standard query engine for the vector index

# Step 6: Define tools for summarization and specific queries
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to The Google PageRank Algorithm",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Get the important concept form the paper",
)

# Step 7: Combine tools into a router query engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),  # Selector to route queries to the appropriate tool
    query_engine_tools=[
        summary_tool,  # Tool for summarization
        vector_tool,  # Tool for specific questions
    ],
    verbose=True  # Enable verbose output for debugging
)

# Step 8: Query the documents using the router query engine
response = query_engine.query("What is the summary of the document?")
print("Summary Response:", str(response))

response = query_engine.query("Who is the author of the paper and when was published?")
print("Author and Date Response:", str(response))

response = query_engine.query("What is about?")
print("Papaer is about:", str(response))