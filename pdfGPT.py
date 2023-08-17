import argparse
from dotenv import load_dotenv
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import pickle
from pdfminer.high_level import extract_text
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


# Parser to handle the model argument.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',
                        type=str,
                        required=True,
                        help=f'The pdf file you want to chat with. Only provide the name of the file that is placed in the pdfs/ directory.')
    parser.add_argument("--local",
                        action="store_true",
                        help="Run a local embedder and chat model in stead of the OpenAI api.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Force recreation of the database.")

    return parser.parse_args().file, parser.parse_args().local, parser.parse_args().force


class Pdf:
    def __init__(self, pdf_file, local, force):
        load_dotenv()
        # Checks for openAI API key.
        if os.environ.get('OPENAI_API_KEY') is None:
            print("Please provide a valid OpenAI API key in the .env file. See .env.example for more information")
            exit(1)
        self.filename = pdf_file.removesuffix(".pdf")
        self.pdf_file = f"pdfs/{pdf_file}"
        self.txt_file = f"txts/{self.filename}.txt"
        self.db_name = f"dbs/{self.filename}.pkl"
        self.chunks_path = f"chunks/{pdf_file}_chunks.index"
        self.local = local

        if not os.path.isfile(self.txt_file):
            print(f"Creating {self.txt_file}...")
            self.parse_pdf()
        if not os.path.isfile(self.db_name) or force:
            print(f"Splitting {self.txt_file}...")
            chunks = self.split_txt()
            print(f"Creating {self.db_name}...")
            self.create_faiss_db(chunks)

        self.load_faiss_db()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if self.local:
            callbacks = [StreamingStdOutCallbackHandler()]
            self.qa_chain = ConversationalRetrievalChain.from_llm(GPT4All (model=os.path.abspath("models/ggml-gpt4all-j-v1.3-groovy.bin"),
                                                                           backend="gptj",
                                                                           max_tokens=1000,
                                                                           callbacks=callbacks),
                                                                  self.db.as_retriever(),
                                                                  memory=memory)
            self.qa_chain = ConversationalRetrievalChain.from_llm(LlamaCpp(model_path=os.path.abspath("models/llama-2-13b.q8_0.bin"),
                                                                           max_tokens=5000,
                                                                           n_ctx=2048,
                                                                           n_gpu_layers=1,
                                                                           f16_kv = True,
                                                                           n_batch = 512,
                                                                           verbose=False, # Info about time taken to generate answer.
                                                                           callbacks=[]),
                                                                  self.db.as_retriever(),
                                                                  memory=memory)
        else:
            self.qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.05),
                                                                  self.db.as_retriever(),
                                                                  memory=memory)


    def run(self):
        print("Starting chat. Type 'q' or 'exit' to quit.")
        while True:
            query = input(f"Chatting with {self.filename}.pdf. Ask your question: ").removesuffix("?")
            if not query:
                continue
            if query == "q" or query == "exit":
                return
            res = self.search(query+"?")
            print(res)
            query = ""


    # Converts pdf file to txt file.
    def parse_pdf(self):
        text = extract_text(self.pdf_file)
        with open(self.txt_file, 'w+') as f:
            f.write(text)


    # Splits txt file into chunks.
    def split_txt(self):
        with open(self.txt_file, "r") as f:
            text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=1250, separator="\n\n")
        chunks = []
        splits = text_splitter.split_text(text)
        chunks.extend(splits)

        return chunks


    # Creates and saves Faiss db from the chunks generated by split_txt().
    def create_faiss_db(self, chunks):
        if self.local:
            print(f"Setting up local embedder using {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
            db = FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                                                                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'})) # model_name="all-MiniLM-L6-v2"
        else:
            db = FAISS.from_texts(chunks, OpenAIEmbeddings())
        faiss.write_index(db.index, self.chunks_path)
        db.index = None
        with open(self.db_name, "wb") as f:
            pickle.dump(db, f)


    # Loads Faiss db into memory.
    def load_faiss_db(self):
        index = faiss.read_index(self.chunks_path)

        with open(self.db_name, "rb") as f:
            db = pickle.load(f)

        db.index = index
        self.db = db


    # Searches for the query in the Faiss db.
    def search(self, query):
        result = self.qa_chain({"question": query})
        return result["answer"]


pdf_file, local, force = parse_args()
chatter = Pdf(pdf_file=pdf_file, local=local, force=force)
chatter.run()
