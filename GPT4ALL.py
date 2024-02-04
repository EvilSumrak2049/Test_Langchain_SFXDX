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



