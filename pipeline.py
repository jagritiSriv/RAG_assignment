import os
import tiktoken
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter,SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context


# Reading the file from data folder
def read_file():
  current_dir = os.getcwd()
  print(f"Reading files from--> {current_dir}")
  data = os.path.join(current_dir, 'data')
  documents = SimpleDirectoryReader(input_dir=data).load_data()
  return documents

# hierarchical way of parsing input data - did not give any better result for the given context , hence continued with sentence splitter
def node_parser():
  node_parser = SimpleNodeParser(
      tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
  )
  return node_parser

def text_splitter():
  text_splitter = SentenceSplitter(
    separator=".",
    chunk_size=1024,
    chunk_overlap=20,
    paragraph_separator="\n\n\n",
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode

  )
  return text_splitter

def prompt():
  prompt_helper = PromptHelper(
    context_window=4000, 
    num_output=256, 
    chunk_overlap_ratio=0.1, 
    chunk_size_limit=None

  )
  return prompt_helper

def llm_set():
  llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256) #setting temprature to 0 to avoid 
  embed_model = OpenAIEmbedding()
  return llm,embed_model

def initial_setup(key):
  os.environ['OPENAI_API_KEY'] = key
  splitter = text_splitter()
  helper = prompt()
  llm,embed_model = llm_set()
  return splitter,helper,llm,embed_model

def context(text_splitter,prompt_helper,llm,embed_model):
  service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser= text_splitter,
    prompt_helper=prompt_helper
    )
  return service_context

def index(documents,service_context):
  index = VectorStoreIndex.from_documents(
    documents, 
    service_context = service_context
    )
  query_engine = index.as_query_engine(service_context=service_context)
  return query_engine


def generate_response(query_engine,question):
  response = query_engine.query(question)
  return response

def main(key,question):
  text_splitter,prompt_helper,llm,embed_model = initial_setup(key)
  print("Initial setup completed")

  ser_context = context(text_splitter,prompt_helper,llm,embed_model)
  document = read_file()
  query_engine = index(document,ser_context)
  response = generate_response(query_engine,question)
  print(response)

  return response