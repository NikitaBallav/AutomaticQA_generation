from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import os 
import csv
from langchain_community.vectorstores import FAISS



os.environ["OPENAI_API_KEY"] = "enter your openAPI key"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set file path
file_path = 'marathiStory.txt'

# Load data from text file
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# Print the content of the file
print(data)

question_gen = data

splitter_ques_gen = TokenTextSplitter(
    model_name='gpt-3.5-turbo',
    chunk_size=200,
    chunk_overlap=70
)

chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

# print the chunks
print("\nCreated Document Objects:")
for doc in document_ques_gen:
    print(doc.page_content)


splitter_ans_gen = TokenTextSplitter(
    model_name='gpt-3.5-turbo',
    chunk_size=200,
    chunk_overlap=70
)

document_answer_gen = splitter_ans_gen.split_documents(
    document_ques_gen
)

llm_ques_gen_pipeline = ChatOpenAI(
    temperature=0.5,
    model="gpt-3.5-turbo"
)

prompt_template = """
You are an expert at creating questions based on the context in Marathi language.
Your task is to generate question answer pairs in Marathi language assuming a Marathi chatbot which is answering user questions regarding any government document in Marathi language on a government portal.
Your goal is to prepare a robust Question and Answer database to train a chatbot in Marathi language.
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will help the users to gain information regarding any government document.
Make sure not to lose any important information.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])


refine_template = ("""
You are an expert at creating questions based on government circulars and documentation.
Your task is to generate question answer pairs in Marathi language assuming a Marathi chatbot which is answering user questions regarding any government document in Marathi language on a government portal.
you are capable of framing all possible questions from the Marathi text below.
We have the option to refine the existing questions.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in Marathi.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["text"],
    template=refine_template,
    )

ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                                chain_type = "refine", 
                                                verbose = True, 
                                                question_prompt=PROMPT_QUESTIONS, 
                                                refine_prompt=REFINE_PROMPT_QUESTIONS)


ques = ques_gen_chain.run(document_ques_gen)

print(ques)


llm_answer_gen = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
chain_type = "stuff"

ques_list = ques.split("\n")

answer_generation_chain = load_qa_chain(llm_answer_gen, chain_type)

# Answer each question and save to a file
# for question in ques_list:
#     print("Question: ", question)
#     answer = answer_generation_chain.run(input_documents=document_answer_gen, question=question)
#     print("Answer: ", answer)
#     print("--------------------------------------------------\n\n")
#     # Save answer to file
#     with open("answers.txt", "a", encoding="utf-8") as f:
#         f.write("Question: " + question + "\n")
#         f.write("Answer: " + answer + "\n")
#         f.write("--------------------------------------------------\n\n")


# Extract the file name without extension
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Define the CSV file name with the corresponding text file name
csv_file_name = f"{file_name}_question_answers.csv"

# Open the CSV file in write mode
with open(csv_file_name, "w", newline="", encoding="utf-8") as csvfile:
    # Define the field names for the CSV
    fieldnames = ["Question", "Answer"]
    # Create a CSV writer object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Write the header row
    writer.writeheader()

    # Answer each question and save to the CSV file
    for question in ques_list:
        print("Question: ", question)
        # Answer each question using the QA chain
        answer = answer_generation_chain.run(input_documents=document_answer_gen, question=question)
        print("Answer: ", answer)
        print("--------------------------------------------------\n\n")
        # Write question and answer to the CSV file
        writer.writerow({"Question": question, "Answer": answer})

print(f"CSV file '{csv_file_name}' has been saved with the question-answer pairs corresponding to the text file.")
