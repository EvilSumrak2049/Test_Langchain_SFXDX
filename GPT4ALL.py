#importing the necessary libraries
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
import argparse
import requests

#define arguments for command line
ap = argparse.ArgumentParser()
ap.add_argument('url',help="URL of PDF file")
ap.add_argument('question',help='question for LLM')
ap.add_argument('mode',help='choose mode: cpu or cuda:0')


args = vars(ap.parse_args())


callback = BaseCallbackManager([StreamingStdOutCallbackHandler()])
url=args['url']

#download and write pdf file
r = requests.get(url, stream=True)

with open('myfile.pdf', 'wb') as f:
    f.write(r.content)

#read the llm GPT4ALL model
llm= GPT4All(model=r'Z:\Projects\Test_SFXDX\testSFXDX\orca-mini-3b-gguf2-q4_0.gguf',callback_manager=callback,verbose=True)


#read our pdf file using Langchain
loader = PyPDFLoader(args['url'])
pdf_data = loader.load()

#Define our db (Faiss) and Embedding
model_kwargs = {'device': f'{args["mode"]}'}
db = FAISS.from_documents(pdf_data, HuggingFaceEmbeddings(model_kwargs=model_kwargs))


#Define our template for llm
template = """Respond to the question based on the context.

Question:
{question}

Context:
{context}"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# Prepare the chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

query = ' '.join(args['question'].split('_'))

contexts_list = db.similarity_search(query, k=1)


context = contexts_list[0].page_content



# Run the chain to generate the answer
response = llm_chain.invoke({'question': query, 'context': context[0:300]})
if response['text'] == '':
    print('No data to answer the question')
else:
    print(response['text'])



