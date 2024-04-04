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
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


os.environ["OPENAI_API_KEY"] = "ai key"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set file path
file_path = 'a142.txt'

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
    temperature=0.3,
    model="gpt-3.5-turbo"
)

prompt_template = """
         साहित्य आणि कागदपत्रांवर आधारित प्रश्न तयार करण्यात तुम्ही तज्ञ आहात.
         चॅटबॉटला प्रशिक्षण देण्यासाठी एक मजबूत प्रश्न आणि उत्तर डेटाबेस तयार करणे हे तुमचे ध्येय आहे.
         तुम्ही खालील मजकुराबद्दल प्रश्न विचारून हे करता:

         ------------
         {text}
         ------------

         असे प्रश्न तयार करा जे वापरकर्त्यांना कोणत्याही सरकारी कायद्याबद्दल माहिती मिळवण्यास मदत करतील.
         कोणतीही महत्वाची माहिती गमावणार नाही याची खात्री करा.

         प्रश्न:
         """

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

# refine_template = ("""
#          सरकारी साहित्य आणि कागदपत्रांवर आधारित प्रश्न तयार करण्यात तुम्ही तज्ञ आहात.
#          तुमचे ध्येय हे आहे की वापरकर्त्याला कोणत्याही सरकारी नियमांशी संबंधित माहिती मिळवण्यात मदत करा.
#          आम्हाला काही सराव प्रश्न काही प्रमाणात प्राप्त झाले आहेत: {existing_answer}.
#          आमच्याकडे विद्यमान प्रश्न परिष्कृत करण्याचा किंवा नवीन जोडण्याचा पर्याय आहे.
#          (फक्त आवश्यक असल्यास) खाली आणखी काही संदर्भांसह.
#          ------------
#          {text}
#          ------------

#          नवीन संदर्भ लक्षात घेता मूळ प्रश्न मराठीत परिष्कृत करा.
#          संदर्भ उपयुक्त नसल्यास, कृपया मूळ प्रश्न प्रदान करा.
#          प्रश्न:
#          """
#          )

# REFINE_PROMPT_QUESTIONS = PromptTemplate(
#     input_variables=["existing_answer", "text"],
#     template=refine_template,
# )

ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline, 
                                      chain_type="refine", 
                                      verbose=True, 
                                      question_prompt=PROMPT_QUESTIONS 
                                      )

# refine_prompt=REFINE_PROMPT_QUESTIONS
ques = ques_gen_chain.run(document_ques_gen)

print(ques)


embeddings = OpenAIEmbeddings()

vector_store = FAISS.from_documents(document_answer_gen, embeddings)

llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

ques_list = ques.split("\n")

answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                      chain_type="stuff", 
                                                      retriever=vector_store.as_retriever())


# Answer each question and save to a file
for question in ques_list:
    print("Question: ", question)
    answer = answer_generation_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\n\n")
    # Save answer to file
    with open("answers.txt", "a", encoding="utf-8") as f:
        f.write("Question: " + question + "\n")
        f.write("Answer: " + answer + "\n")
        f.write("--------------------------------------------------\n\n")

