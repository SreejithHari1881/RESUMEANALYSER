import os
import json
import pandas as pd
import traceback
import langchain_openai
import langchain_community
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from src.ResumeAnalyser.utils import read_file
from src.ResumeAnalyser.logger import logging

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2

KEY= os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key= KEY, model_name= "gpt-3.5-turbo", temperature= 1)

resume_analysis_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
    You are an AI specializing in analyzing professional resumes. Given the text from a resume below, provide a structured analysis containing:
    
    1. **Summary**: A brief overview of the candidate's profile.
    2. **Skills**: List the main skills mentioned, categorized by technical and non-technical.
    3. **Experience**: Summarize key work experiences, highlighting roles, responsibilities, and achievements.
    4. **Education**: Outline the candidate's educational background.
    5. **Certifications**: List any certifications if present.
    6. **Relevant Technologies**: Mention any technologies, frameworks, or programming languages.
    7. **Professional Keywords**: Identify any industry-specific keywords that may be relevant to the candidate's field.
    
    Input Resume:
    {resume_text}
    
    Output:
    - **Summary**: ...
    - **Skills**: ...
    - **Experience**: ...
    - **Education**: ...
    - **Certifications**: ...
    - **Relevant Technologies**: ...
    - **Professional Keywords**: ...
    """,
)

resume_analysis_chain = LLMChain(llm=llm, prompt= resume_analysis_prompt, output_key="analysis_output", verbose=True)

resume_evaluation_prompt = PromptTemplate(
    input_variables=["resume_text", "target_job_role"],
    template="""
    You are an AI specializing in professional resume evaluation and improvement. Given the resume text and the target job role, perform the following:

    1. **Evaluation**:
       - Identify and evaluate the strengths in the resume.
       - Identify weaknesses or gaps that may hinder the candidate's chance for the target job role.
    
    2. **Improvement Suggestions**:
       - Suggest specific improvements for each identified weakness or gap.
       - Recommend ways to make the resume more concise, impactful, or targeted for the desired role.
       - If applicable, provide keywords or phrases commonly used in the industry that would align with the target job role.
       - Recommend any additional skills, certifications, or experience that would make the candidate more competitive.

    Target Job Role:
    {target_job_role}

    Resume:
    {resume_text}

    Output:
    - **Evaluation**:
      - Strengths: ...
      - Weaknesses: ...
    
    - **Improvement Suggestions**:
      - Suggested Improvements: ...
      - Additional Keywords/Skills: ...
    """
)

resume_evaluation_chain = LLMChain(llm=llm, prompt= resume_evaluation_prompt, output_key="evaluation_output", verbose=True)

generate_chain = SequentialChain(chains=[resume_analysis_chain,resume_evaluation_chain], 
                                 input_variables=['resume_text',"target_job_role"],
                                 output_variables=["analysis_output", "evaluation_output"],
                                 verbose=True
                                 )

file_path= r"C:\Users\91897\RESUMEANALYSER\data.txt"

with open(file_path, 'r') as file:
    resume_text= file.read()

target_job_role = ["Test_manager","QA Manager"]

with get_openai_callback() as cb:
    response = generate_chain(
        {
            'resume_text': resume_text,
            'target_job_role': target_job_role,
            'response_json': response_json
            
        }
    )

print(f"Total tokens:{cb.total_tokens}")
print(f"Prompt tokens:{cb.prompt_tokens}")
print(f"Completion tokens:{cb.completion_tokens}")
print(f"Total cost:{cb.total_cost}") 

json_response = json.dumps(response, indent=2)
data = json.loads(json_response)

analysis_output = data.get("analysis_output", "")
evaluation_output = data.get("evaluation_output", "")

print("Analysis Output:")
print(analysis_output)

print("\nEvaluation Output:")
print(evaluation_output)








