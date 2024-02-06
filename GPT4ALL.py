from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
import argparse

#define arguments for command line
ap = argparse.ArgumentParser()
ap.add_argument('url',help="URL of PDF file")
ap.add_argument('question',help='question for LLM')
ap.add_argument('mode',help='choose mode: cpu or gpu')


args = vars(ap.parse_args())


callback = BaseCallbackManager([StreamingStdOutCallbackHandler()])


#read the llm GPT4ALL model
try:
    llm= GPT4All(model=r'./orca-mini-3b-gguf2-q4_0.gguf',callback_manager=callback,verbose=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


#read our pdf file using Langchain
try:
    loader = PyPDFLoader(args['url'])
    pdf_data = loader.load()
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit()


text_splitter = RecursiveCharacterTextSplitter(
   chunk_size = 256,
   chunk_overlap  = 0,
   length_function = len)


try:
    texts = text_splitter.split_documents(pdf_data)
except Exception as e:
    print(f"Error splitting PDF: {e}")
    exit()




mode = 'cpu' if args['mode'] == 'cpu' else 'cuda:0'
model_kwargs = {'device': mode}


#Define our db (Faiss) and Embedding
try:
    db = FAISS.from_documents(texts, HuggingFaceEmbeddings(model_kwargs=model_kwargs))
except Exception as e:
    print(f"Error creating FAISS index: {e}")
    exit()


#Define our template for llm
template = """Respond to the question based on the context.
Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)





query = args['question']

try:
    contexts_list = db.similarity_search_with_score(query,k=10)
except Exception as e:
    print(f"Error searching for similar documents: {e}")
    exit()


context = '\n'.join([doc.page_content for doc, score in contexts_list if score<1.35])




# Prepare the chain and run the chain to generate the answer
if context == '':
    print('No data to answer the question')
else:
    chain = (
            prompt
            | llm
            | StrOutputParser()
    )
    try:
        response = chain.invoke({'question': query, 'context': context})
    except Exception as e:
        print(f"Error generating response: {e}")
        exit()








