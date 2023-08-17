# pdfGPT
Query the different UVA Computer Science courses using Faiss. This tool allows you to chat with different CS textbooks using Langchain and the OpenAI api. A chat history memory is kept, so asking follow up questions is supported.
When the answer to the asked question is not to be found in the document, the model responds with "I don't know".

# Usage
To use pdfGPT, first download the requirements by running:
```
python3 -m pip install -r requirements.txt
```
Next, you need an [OpenAI api key](https://platform.openai.com/overview). Add this key to your .env file, and you can start the client using:
```
python3 pdfGPT.py -f <FILENAME>
```
or
```
python3 pdfGPT.py --file <FILENAME>
```
the filename that is to passed should be the name of the pdf file, including suffix and excluding the path to the file.

If you want to add a custom model, simply place the pdf- or txt file in its respective directory and run the program. It will automatically create a Faiss database that is allows lookup of questions.

# Demo
.
