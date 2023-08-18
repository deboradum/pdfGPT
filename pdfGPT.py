import argparse
from dotenv import load_dotenv
import faiss
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import pickle
from pdfminer.high_level import extract_text
import torch
from pprint import pprint


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
    parser.add_argument("--save",
                        action="store_true",
                        help="Save chat history, including the source of the answer to a local json file.")

    file_name = parser.parse_args().file
    local = parser.parse_args().local
    force = parser.parse_args().force
    save = parser.parse_args().save
    return file_name, local, force, save


class Pdf:
    def __init__(self, pdf_file, local, force, save_convo):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        load_dotenv()
        # Checks for openAI api key.
        if os.environ.get('OPENAI_API_KEY') is None and not local:
            print("Please provide a valid OpenAI api key in the .env file. See .env.example for more information")
            raise Exception("No OpenAI api key")

        self.filename = pdf_file.removesuffix(".pdf")
        self.pdf_file = f"pdfs/{pdf_file}"
        self.txt_file = f"txts/{self.filename}.txt"
        self.db_name = f"dbs/{self.filename}.pkl"
        self.chunks_path = f"chunks/{pdf_file}_chunks.index"
        self.local = local
        self.save_convo = save_convo
        self.save_path = None

        if not os.path.isfile(self.txt_file):
            print(f"Creating {self.txt_file}...")
            self.parse_pdf()
        if not os.path.isfile(self.db_name) or force:
            print(f"Splitting {self.txt_file}...")
            chunks = self.split_txt()
            print(f"Creating {self.db_name}...")
            self.create_faiss_db(chunks)

        self.load_faiss_db()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,
                                          input_key='question', output_key='answer')
        chatTemplate = """
                       Answer the question based on the chat history(delimited by <hs></hs>) and context(delimited by <ctx> </ctx>) below. If you don't know the answer based on the given context, just say that you don't know, don't try to make up an answer. Keep the answers as concise as possible, but do explain your answer by quoting a piece of the context.
                       -----------
                       <ctx>
                       {context}
                       </ctx>
                       -----------
                       <hs>
                       {chat_history}
                       </hs>
                       -----------
                       Question: {question}
                       Answer:
                       """
        templatePrompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=chatTemplate)
        llm = self.setup_llm()
        self.qa_chain = ConversationalRetrievalChain.from_llm(llm,
                                                              self.db.as_retriever(),
                                                              memory=memory,
                                                              return_source_documents=self.save_convo,
                                                              combine_docs_chain_kwargs={'prompt': templatePrompt} if local else None)


    # Sets up the llm, depending on the use of the '--local' flag.
    def setup_llm(self):
        callbacks = []
        if self.local:
            if os.environ.get('MODEL_PATH') is None or not os.path.isfile(os.environ.get('MODEL_PATH')):
                print("Please specify a valid model path that is to be used for local inference.")
                raise Exception("No model specified")
            return LlamaCpp(model_path=os.path.abspath(os.environ.get('MODEL_PATH')),
                            max_tokens=8192,
                            n_ctx=2048,
                            n_gpu_layers=1,
                            f16_kv = True,
                            n_batch = 512,
                            verbose=False, # Info about time taken to generate answer.
                            callbacks=callbacks)

        return OpenAI(temperature=0.05)

    def run(self):
        print("Starting chat. Type 'q' or 'exit' to quit.")
        while True:
            query = input(f"Chatting with {self.filename}.pdf. Ask your question: ")
            if not query or query.strip() == "":
                continue
            if query == "q" or query == "exit":
                self.finish_save_file()
                return
            res = self.search(query)
            ans, src = res["answer"], [] if not self.save_convo else res['source_documents']
            print(f"\t{ans}")
            if self.save_convo:
                self.save_conversation(query, ans, src)
            query = ""


    # Saves the conversation by writing to a json file.
    def save_conversation(self, query, answer, sources):
        if self.save_path is None:
            i = len([f for f in os.listdir("conversations/") if f.startswith(self.filename)])
            self.save_path = f"conversations/{self.filename}-conversation{i:02d}.json"
            with open(self.save_path, "a") as f:
                conv = {"question": query, "answer": answer, "sources": [doc.page_content for doc in sources]}
                f.write("[" + json.dumps(conv))
            return

        with open(self.save_path, "a") as f:
            conv = {"question": query, "answer": answer, "sources": [doc.page_content for doc in sources]}
            f.write("," + json.dumps(conv))


    # Closes the json array.
    def finish_save_file(self):
        if self.save_path is None:
            return
        with open(self.save_path, "a") as f:
            f.write("]")


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
        return self.qa_chain({"question": query})


pdf_file, local, force, save = parse_args()
chatter = Pdf(pdf_file=pdf_file, local=local, force=force, save_convo=save)
chatter.run()
