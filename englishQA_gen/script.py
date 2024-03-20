from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import os 
from langchain_community.vectorstores import FAISS
import csv


os.environ["OPENAI_API_KEY"] = "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"

# Define the directory containing the PDF files
directory = 'C:\chatbot nic\engdata\english8'

# List all files in the directory
files = os.listdir(directory)

# Process each file
for file_name in files:
    if file_name.endswith('.pdf'):  # Check if the file is a PDF
        file_path = os.path.join(directory, file_name)
        
        # Load data from PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()

        question_gen = ''


        for page in data:
            question_gen += page.page_content
            
        splitter_ques_gen = TokenTextSplitter(
            model_name = 'gpt-3.5-turbo',
            chunk_size = 10000,
            chunk_overlap = 200
        )

        chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

        document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

        splitter_ans_gen = TokenTextSplitter(
            model_name = 'gpt-3.5-turbo',
            chunk_size = 1000,
            chunk_overlap = 100
        )


        document_answer_gen = splitter_ans_gen.split_documents(
            document_ques_gen
        )

        llm_ques_gen_pipeline = ChatOpenAI(
            temperature = 0.3,
            model = "gpt-3.5-turbo"
        )

        prompt_template = """
        You are an expert at creating questions based on materials and documentation.
        Your goal is to prepare a robust Question and answer database to train a chatbot.
        You do this by asking questions about the text below:

        ------------
        {text}
        ------------

        Create questions that will help the users to gain information regarding any government act.
        Make sure not to lose any important information.

        QUESTIONS:
        """

        PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

        refine_template = ("""
        You are an expert at creating questions based on government circulars and documentation.
        Your goal is to help a user to gain information regarding any government regulations.
        We have received some practice questions to a certain extent: {existing_answer}.
        We have the option to refine the existing questions or add new ones.
        (only if necessary) with some more context below.
        ------------
        {text}
        ------------

        Given the new context, refine the original questions in English.
        If the context is not helpful, please provide the original questions.
        QUESTIONS:
        """
        )

        REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )

        ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                                chain_type = "refine", 
                                                verbose = True, 
                                                question_prompt=PROMPT_QUESTIONS, 
                                                refine_prompt=REFINE_PROMPT_QUESTIONS)

        ques = ques_gen_chain.run(document_ques_gen)

        print(ques)


        embeddings = OpenAIEmbeddings()

        vector_store = FAISS.from_documents(document_answer_gen, embeddings)

        llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

        ques_list = ques.split("\n")

        ques_list

        answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                    chain_type="stuff", 
                                                    retriever=vector_store.as_retriever())


       # Initialize list to store question-answer pairs
        qa_pairs = []

        # Answer each question and save to a list
        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")
            qa_pairs.append({'Question': question, 'Answer': answer})

        # Write question-answer pairs to a CSV file in the same directory
        csv_file_name = file_name.replace('.pdf', '_qa_pairs1.csv')
        csv_file_path = os.path.join(directory, csv_file_name)
        fieldnames = ['Question', 'Answer']

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qa_pairs)

        print("Question-answer pairs saved to:", csv_file_path)