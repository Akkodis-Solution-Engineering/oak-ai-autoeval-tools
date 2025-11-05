import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import plotly.express as px
import numpy as np
import json
import re
from openai import OpenAI, AzureOpenAI
from utils.formatting import *
import plotly.graph_objects as go
from utils.db_scripts import get_db_connection, insert_single_lesson_plan
from utils.common_utils import  log_message, get_env_variable
from utils.constants import ErrorMessages
import requests

# Load environment variables
load_dotenv()



def execute_single_query(query, params):
    try:
        connection = get_db_connection()  # Assuming this function gets a database connection
        cursor = connection.cursor()
        cursor.execute(query, params)
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        log_message("error", f"Unexpected error executing query: {e}")
        return False
    

def fetch_lesson_plan_sets(limit=None):
    """
    Fetch the contents of the lesson_plan_sets table and load into a pandas DataFrame.

    Args:
        limit (int or None): The maximum number of rows to retrieve. If None or 0, fetch all rows.

    Returns:
        pd.DataFrame: DataFrame containing the lesson_plan_sets data.
    """
    try:
        conn = get_db_connection()  # Assuming this is a function that returns a connection object
        if limit and limit > 0:
            query = "SELECT * FROM lesson_plan_sets LIMIT %s;"
            df = pd.read_sql_query(query, conn, params=[limit])
        else:
            query = "SELECT * FROM lesson_plan_sets;"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def fetch_sample_sets(limit=5):
    """
    Fetch the contents of the lesson_plan_sets table and load into a pandas DataFrame.

    Args:
        limit (int or None): The maximum number of rows to retrieve. If None or 0, fetch all rows.

    Returns:
        pd.DataFrame: DataFrame containing the lesson_plan_sets data.
    """
    try:
        conn = get_db_connection()  # Assuming this is a function that returns a connection object
        if limit and limit > 0:
            query = """SELECT DISTINCT ON (subject)
                            lesson_number, 
                            subject, 
                            key_stage, 
                            lesson_title
                        FROM public.lesson_plan_sets
                        ORDER BY subject, key_stage, lesson_number LIMIT %s;"""
            df = pd.read_sql_query(query, conn, params=[limit])
        else:
            query = """SELECT DISTINCT ON (subject)
                            lesson_number, 
                            subject, 
                            key_stage, 
                            lesson_title
                        FROM public.lesson_plan_sets
                        ORDER BY subject, key_stage, lesson_number;"""
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define the clean_response function
def clean_response(content):
    try:
        # Assuming content is a JSON string, try to parse it
        content_json = json.loads(content)
        status = "SUCCESS" if content_json else "FAILURE"
        return content_json, status
    except json.JSONDecodeError:
        return content, "FAILURE"




def run_agent_openai_inference(prompt, llm_model, llm_model_temp, top_p=1, timeout=150):
    """Run inference using OpenAI or Azure OpenAI based on the model selection.

    Args:
        prompt: The prompt to send to the model
        llm_model: Model name (use 'azure-openai' for Azure OpenAI)
        llm_model_temp: Temperature parameter
        top_p: Top-p parameter
        timeout: Request timeout in seconds

    Returns:
        Dict with 'response' key containing the model output
    """
    # Initialize the appropriate client based on model selection
    if llm_model.startswith("azure-") or llm_model.lower() == "azure-openai":
        # Initialize Azure OpenAI client
        api_key = get_env_variable("AZURE_OPENAI_API_KEY")
        endpoint = get_env_variable("AZURE_OPENAI_ENDPOINT")
        api_version = get_env_variable("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        deployment_name = get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME")

        log_message("info", f"Using Azure OpenAI deployment: {deployment_name}")

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            timeout=timeout
        )
        model_for_request = deployment_name
    else:
        # Standard OpenAI client
        api_key = get_env_variable("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, timeout=timeout)
        model_for_request = llm_model

    try:
        response = client.chat.completions.create(
            model=model_for_request,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            seed=42,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        message = response.choices[0].message.content
        cleaned_content, status = clean_response(message)
        return {
            "response": cleaned_content
        }

    except Exception as e:
        log_message("error", f"Unexpected error during inference: {e}")
        return {
            "response": {
                "result": None,
                "justification": f"An error occurred: {e}",
            },
            "status": "FAILURE",
        }
    
selection = st.selectbox('Select a lesson plan set to generate lesson plans with:', ['HB_Test_Set','Model_Compare_Set_10'])
# Fetch the data and load it into a DataFrame

if selection == 'HB_Test_Set':
    lessons_df = fetch_lesson_plan_sets(0)
    lessons_df['key_stage'] = lessons_df['key_stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])

    st.write(lessons_df)
elif selection == 'Model_Compare_Set_10':
    lessons_df = fetch_sample_sets(5)
    lessons_df['key_stage'] = lessons_df['key_stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])

    st.write(lessons_df)
else:
    st.error("Invalid selection. Please select a valid lesson plan set.")





if 'llm_model' not in st.session_state: 
    st.session_state.llm_model = 'gpt-4o-2024-05-13'
if 'llm_model_temp' not in st.session_state:
    st.session_state.llm_model_temp = 0.1


llm_model_options = ['o1-preview-2024-09-12','o1-mini-2024-09-12','gpt-4o-mini-2024-07-18', "gpt-4o",
    "gpt-4o-mini",'gpt-4o-2024-05-13','gpt-4o-2024-08-06','chatgpt-4o-latest',
                     'gpt-4-turbo-2024-04-09','gpt-4-0125-preview','gpt-4-1106-preview','azure-openai']

# Get default value, ensuring it's in the options list
default_model = st.session_state.llm_model
if isinstance(default_model, str):
    # If it's a single string, convert to list
    default_list = [default_model] if default_model in llm_model_options else ['gpt-4o-2024-05-13']
else:
    # If it's already a list, filter to only include valid options
    default_list = [m for m in default_model if m in llm_model_options]
    if not default_list:
        default_list = ['gpt-4o-2024-05-13']

st.session_state.llm_model = st.multiselect(
    'Select models for lesson plan generation:',
    llm_model_options,
    default=default_list
)
st.session_state.llm_model

# todo: add number of lesson plans that will be generated for each model 



st.session_state.llm_model_temp = st.number_input(
    'Enter temperature for the model:',
    min_value=0.0, max_value=2.00,
    value=st.session_state.llm_model_temp,
    help='Minimum value is 0.0, maximum value is 2.00.'
)

response = None

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current script's directory
base_dir = os.path.dirname(script_dir)

# Define the file path for prompt_raw.txt in the data directory
prompt_file_path = os.path.join(base_dir, 'data', 'big_lp_generator_prompt.txt')


# Check if the file exists
if not os.path.exists(prompt_file_path):
    st.error(f"File not found: {prompt_file_path}")
else:
    # Read the prompt from data/prompt_raw.txt
    with open(prompt_file_path, 'r') as file:
        prompt_template = file.read()

    st.write('Review the Prompt for generations')
    with st.expander("Prompt Template", expanded=False):
        st.text_area("Generation Prompt", prompt_template, height=600)

llm_models = st.session_state.llm_model  # This will be a list of selected models from the multiselect
llm_model_temp = st.session_state.llm_model_temp


if 'top_p' not in st.session_state:
    st.session_state.top_p = 1.0  # Ensure this is a float


st.session_state.top_p = st.number_input(
    'Enter top_p for the model:',
    min_value=0.0, max_value=1.0,  # These should be floats
    value=float(st.session_state.top_p),  # Convert value to float
    step=0.01,  # You may need to specify the step value, e.g., 0.01
    help='Minimum value is 0.0, maximum value is 1.00.'
)

# Usage in Streamlit form
with st.form(key='generation_form'):
    if st.form_submit_button('Start Generation'):
        for llm_model in llm_models:
            for index, row in lessons_df.iterrows():
                # Replace placeholders with actual values in the prompt
                prompt = prompt_template.replace("{{key_stage}}", row['key_stage'])
                prompt = prompt.replace("{{subject}}", row['subject'])
                prompt = prompt.replace("{{lesson_title}}", row['lesson_title'])

                
                response = run_agent_openai_inference(prompt, llm_model, llm_model_temp,st.session_state.top_p)
                

                st.write(f"**Response for {row['key_stage']} - {row['subject']} - {row['lesson_title']}**")
                st.write(f"Model: `{llm_model}`")

                # Extract the 'response' field from the API response
                response = response['response']

                # Display the response in the UI in a friendly format
                with st.expander("📖 View Generated Lesson Plan", expanded=True):
                    if isinstance(response, dict):
                        # Display title and basic info
                        st.markdown(f"### {response.get('title', 'Lesson Plan')}")
                        st.markdown(f"**Key Stage:** {response.get('keyStage', 'N/A')}")
                        st.markdown(f"**Subject:** {response.get('subject', 'N/A')}")
                        st.markdown(f"**Topic:** {response.get('topic', 'N/A')}")

                        # Learning Outcome
                        st.markdown("---")
                        st.markdown("#### 🎯 Learning Outcome")
                        st.info(response.get('learningOutcome', 'N/A'))

                        # Learning Cycles
                        if 'learningCycles' in response and response['learningCycles']:
                            st.markdown("#### 📚 Learning Cycles")
                            for i, cycle in enumerate(response['learningCycles'], 1):
                                st.markdown(f"{i}. {cycle}")

                        # Prior Knowledge
                        if 'priorKnowledge' in response and response['priorKnowledge']:
                            st.markdown("#### 📝 Prior Knowledge")
                            for knowledge in response['priorKnowledge']:
                                st.markdown(f"- {knowledge}")

                        # Key Learning Points
                        if 'keyLearningPoints' in response and response['keyLearningPoints']:
                            st.markdown("#### ✨ Key Learning Points")
                            for point in response['keyLearningPoints']:
                                st.markdown(f"- {point}")

                        # Keywords
                        if 'keywords' in response and response['keywords']:
                            st.markdown("#### 📖 Keywords")
                            for kw in response['keywords']:
                                st.markdown(f"**{kw.get('keyword', '')}:** {kw.get('definition', '')}")

                        # Misconceptions
                        if 'misconceptions' in response and response['misconceptions']:
                            st.markdown("#### ⚠️ Common Misconceptions")
                            for misc in response['misconceptions']:
                                st.warning(f"**{misc.get('misconception', '')}**\n\n{misc.get('description', '')}")

                        # Starter Quiz
                        if 'starterQuiz' in response and response['starterQuiz']:
                            st.markdown("#### 🎲 Starter Quiz")
                            for i, q in enumerate(response['starterQuiz'], 1):
                                st.markdown(f"**Q{i}: {q.get('question', '')}**")
                                st.markdown(f"- ✅ Answer: {', '.join(q.get('answers', []))}")
                                st.markdown(f"- ❌ Distractors: {', '.join(q.get('distractors', []))}")

                        # Learning Cycles Details
                        for cycle_num in ['cycle1', 'cycle2', 'cycle3']:
                            if cycle_num in response:
                                cycle_data = response[cycle_num]
                                st.markdown("---")
                                st.markdown(f"#### 🔄 {cycle_data.get('title', f'Cycle {cycle_num[-1]}')} ({cycle_data.get('durationInMinutes', 0)} mins)")

                                if 'explanation' in cycle_data:
                                    exp = cycle_data['explanation']
                                    st.markdown("**Explanation:**")
                                    st.markdown(exp.get('spokenExplanation', ''))
                                    if exp.get('slideText'):
                                        st.info(f"💡 {exp.get('slideText')}")

                                if 'checkForUnderstanding' in cycle_data:
                                    st.markdown("**Check for Understanding:**")
                                    for check in cycle_data['checkForUnderstanding']:
                                        st.markdown(f"- Q: {check.get('question', '')}")
                                        st.markdown(f"  - Answer: {', '.join(check.get('answers', []))}")

                                if 'practice' in cycle_data:
                                    st.markdown(f"**Practice:** {cycle_data['practice']}")

                                if 'feedback' in cycle_data:
                                    st.success(f"**Feedback:** {cycle_data['feedback']}")

                        # Exit Quiz
                        if 'exitQuiz' in response and response['exitQuiz']:
                            st.markdown("---")
                            st.markdown("#### 🎯 Exit Quiz")
                            for i, q in enumerate(response['exitQuiz'], 1):
                                st.markdown(f"**Q{i}: {q.get('question', '')}**")
                                st.markdown(f"- ✅ Answer: {', '.join(q.get('answers', []))}")
                                st.markdown(f"- ❌ Distractors: {', '.join(q.get('distractors', []))}")

                        # Additional Materials
                        if 'additionalMaterials' in response:
                            st.markdown("---")
                            st.markdown("#### 📦 Additional Materials")
                            st.markdown(response['additionalMaterials'])

                        # Divider before JSON
                        st.markdown("---")
                        st.markdown("##### 🔍 Raw JSON Data")
                        st.json(response)
                    else:
                        # Fallback to JSON display
                        st.json(response)

                # Display raw JSON outside the main expander for easy access
                with st.expander("🔍 View Raw JSON Only", expanded=False):
                    st.json(response)

                # Convert the response to a JSON string
                response = json.dumps(response)

                # Clean up the response by removing escape characters and line breaks
                response_cleaned = re.sub(r'\\n|\\r', '', response)

                lesson_id = selection +'_'+ str(row['lesson_number'])+'_'+ 'gpt-4o_Comparison_Set'
                # st.write(f'Lesson ID: {lesson_id}')
                # st.write(f'llm_model: {llm_model}')
                # st.write(f'llm_model_temp: {llm_model_temp}')
                # st.write(f'top_p: {st.session_state.top_p}')
                # st.write(f"Selection: {selection}")
                generation_details_value = llm_model + '_' + str(llm_model_temp) + '_' + selection + '_' + str(st.session_state.top_p)
                st.write(f"Generation Details: {generation_details_value}")
                # Insert the generated lesson plan into the database
                lesson_plan_id = insert_single_lesson_plan(response_cleaned,lesson_id, row['key_stage'], row['subject'],  generation_details_value)
                # Display the lesson plan ID in the Streamlit app
                st.write(f"Lesson Plan ID: {lesson_plan_id}")