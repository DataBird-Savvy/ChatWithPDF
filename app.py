import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from styles import css, bot_template, user_template

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFChatbot:
    def __init__(self):
        self.api_key = None
        self.embeddings = None
        self.vectorstore = None
        self.chain = None
        self.init_environment()
    
    def init_environment(self):
        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def get_pdf_text(self, pdf_docs):
        logger.info("Starting PDF text extraction.")
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.info("PDF text extraction completed.")
        return text

    def get_text_chunks(self, text):
        logger.info("Starting text chunking.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        logger.info("Text chunking completed.")
        return chunks

    def get_vectorstore(self, text_chunks):
        logger.info("Creating vector store.")
        self.vectorstore = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        self.vectorstore.save_local("faiss_index")
        logger.info("Vector store created and saved.")
        return self.vectorstore

    def get_conversation_chain(self):
        logger.info("Initializing conversation chain.")
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        logger.info("Conversation chain initialized.")
        return self.chain

    def handle_userinput(self, user_question):
        logger.info("Handling user input.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        try:
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = self.get_conversation_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            # Log and check the response
            logger.debug(f"Response: {response}")
            
            # Extract output_text from response
            output_text = response.get('output_text', 'No response generated.')

            # Append user question and bot response to chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            st.session_state.chat_history.append({'content': user_question, 'role': 'user'})
            st.session_state.chat_history.append({'content': output_text, 'role': 'bot'})
            
            # Display chat history
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    content = message.get('content', '')
                    role = message.get('role', '')
                    if role == 'user':
                        st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
                    elif role == 'bot':
                        st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)
            else:
                st.write("No chat history found.")
            
            logger.info("User input handled successfully.")
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            st.error(f"Error handling user input: {e}")

    def process_pdfs(self, pdf_docs):
        with st.spinner("Processing"):
            try:
                raw_text = self.get_pdf_text(pdf_docs)
                text_chunks = self.get_text_chunks(raw_text)
                self.get_vectorstore(text_chunks)
                self.get_conversation_chain()
                logger.info("Processing completed successfully.")
            except Exception as e:
                logger.error(f"Error during processing: {e}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    chatbot = PDFChatbot()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    
    # Input and process PDFs
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                chatbot.process_pdfs(pdf_docs)
            else:
                st.warning("Please upload PDF documents before processing.")

    # Input question and process it
    user_question = st.text_input("Ask a question about your documents:")
    if st.button("Ask"):
        if user_question:
            chatbot.handle_userinput(user_question)
        else:
            st.warning("Please enter a question before submitting.")

if __name__ == '__main__':
    main()
