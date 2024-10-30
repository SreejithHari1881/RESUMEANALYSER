import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import PyPDF2
from io import StringIO

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("🎯 Professional Resume Analyzer")
st.markdown("""
This app analyzes your resume and provides detailed feedback based on your target job role.
Upload your resume (PDF or TXT) and get instant insights!
""")

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text()
            return resume_text
            
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
            return None
        
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
        return text
    
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
        return None

def initialize_llm():
    try:
        KEY = os.getenv("OPENAI_API_KEY")
        if not KEY:
            st.error("OpenAI API key not found. Please check your .env file.")
            return None
        
        return ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=1)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def create_chains(llm):
    # Resume Analysis Prompt
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
        """
    )

    # Resume Evaluation Prompt
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

        Target Job Role: {target_job_role}
        Resume: {resume_text}
        """
    )

    # Create chains
    analysis_chain = LLMChain(llm=llm, prompt=resume_analysis_prompt, output_key="analysis_output")
    evaluation_chain = LLMChain(llm=llm, prompt=resume_evaluation_prompt, output_key="evaluation_output")
    
    return SequentialChain(
        chains=[analysis_chain, evaluation_chain],
        input_variables=['resume_text', 'target_job_role'],
        output_variables=['analysis_output', 'evaluation_output']
    )

def main():
    # File upload
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'txt'])
    
    # Job role input
    target_job_role = st.text_input("Enter your target job role", "")

    if uploaded_file and target_job_role:
        with st.spinner("Reading file..."):
            resume_text = read_file(uploaded_file)
            
        if resume_text:
            llm = initialize_llm()
            if llm:
                generate_chain = create_chains(llm)
                
                with st.spinner("Analyzing resume..."):
                    try:
                        with get_openai_callback() as cb:
                            response = generate_chain({
                                'resume_text': resume_text,
                                'target_job_role': target_job_role
                            })
                        
                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🎯 Evaluation", "📈 Usage Stats"])
                        
                        with tab1:
                            st.markdown("### Resume Analysis")
                            st.markdown(response['analysis_output'])
                        
                        with tab2:
                            st.markdown("### Resume Evaluation")
                            st.markdown(response['evaluation_output'])
                        
                        with tab3:
                            st.markdown("### Token Usage Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Tokens", cb.total_tokens)
                            with col2:
                                st.metric("Prompt Tokens", cb.prompt_tokens)
                            with col3:
                                st.metric("Completion Tokens", cb.completion_tokens)
                            with col4:
                                st.metric("Total Cost ($)", round(cb.total_cost, 4))
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()