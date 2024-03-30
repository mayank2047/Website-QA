from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.1)

#llm=GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'], temprature=0.1)


instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path="C:/Users/DELL/Desktop/codabasic_Q&A/FAISS INDEX"

def create_vector_db():
    
    
    loader = CSVLoader(file_path="C:/Users/DELL/Desktop/codabasic_Q&A/codebasics_faqs.csv", source_column='prompt', encoding='latin-1')
    data = loader.load()
           

            


    #data = loader.load()

    vectordb=FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # load the vector database from the local folder
    vectordb= FAISS.load_local(vectordb_file_path,instructor_embeddings, allow_dangerous_deserialization=True)

    #Create a retriever for querying the vector database 
    retriever=vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context.
    if the answer is not found in the context , kindly state "I don't know." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT= PromptTemplate(
        template=prompt_template,input_variables=["context","question"]

    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",return_source_documents=True,
                chain_type_kwargs={"prompt":PROMPT})
    return chain

if __name__ == "__main__":
    chain= get_qa_chain()

    print(chain("do you provide intership and do you  have EMI option "))