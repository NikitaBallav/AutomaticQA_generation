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
file_path = 'a142.pdf'

# Load data from PDF
loader = PyPDFLoader(file_path)
data = loader.load()

question_gen = ''

for page in data:
    question_gen += page.page_content

splitter_ques_gen = TokenTextSplitter(
    model_name='gpt-3.5-turbo',
    chunk_size=1000,
    chunk_overlap=100
)
# splitter_ques_gen : used to split text into chunks or segments.
# These chunks are stored in the variable chunks_ques_gen.

chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
# Print the number of chunks generated
print(f"Number of Chunks Generated: {len(chunks_ques_gen)}")

document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
# This suggests that each chunk is being treated as a separate document

splitter_ans_gen = TokenTextSplitter(
    model_name='gpt-3.5-turbo',
    chunk_size=1000,
    chunk_overlap=100
)

document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

llm_ques_gen_pipeline = ChatOpenAI(
    temperature=0.3,
    model="gpt-3.5-turbo"
)

prompt_template = """
        आपण साहित्य आणि कागदपत्रांवर आधारित प्रश्न तयार करण्यात निपुण आहात.
        आपला उद्दिष्ट एक चॅटबॉट प्रशिक्षित करण्यासाठी दृढ प्रश्न आणि उत्तर डेटाबेस तयार करणे आहे.
        आपल्याला खालील मजकूर विशेषज्ञांच्या विषयी प्रश्न प्रस्तुत करायचे आहे:
        
        ------------
        {text}
        ------------

        सरकारचे कोणत्याही कायद्यांबद्दल माहिती मिळवण्यासाठी प्रश्न तयार करा.
        महत्त्वाची कोणतीही माहिती गमू नये हे सुनिश्चित करा.

        प्रश्न:
        """

refine_template = ("""
        आपण सरकारी साहित्य आणि कागदपत्रांवर आधारित प्रश्न तयार करण्यात निपुण आहात.
        आपला उद्दिष्ट एक वापरकर्त्याला कोणत्याही सरकारी नियमांबद्दल माहिती मिळवण्यास मदत करणे आहे.
        आम्ही काही अभ्यास प्रश्ने प्राप्त केल्या आहेत आणि ते काही प्रमाणात आहेत: {existing_answer}.
        आम्हाला विकसित करण्याचा पर्याय आहे किंवा नवीन प्रश्न जोडण्याचा.
        (आवश्यक असल्यास) काही अधिक संदर्भाशी खात्री.
        ------------
        {text}
        ------------

        नवीन संदर्भ दिल्यानुसार, मूळ प्रश्नांचे संशोधन करा.
        जर संदर्भ मदतगार नसेल तर, कृपया मूळ प्रश्न प्रदान करा.
        
        प्रश्न:
        """
)



PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

REFINE_PROMPT_QUESTIONS = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template)

ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline, 
                                      chain_type="refine", 
                                      verbose=True, 
                                      question_prompt=PROMPT_QUESTIONS,
                                      refine_prompt=REFINE_PROMPT_QUESTIONS
                                      )


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

