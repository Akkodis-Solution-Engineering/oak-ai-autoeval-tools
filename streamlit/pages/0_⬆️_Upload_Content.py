"""
Streamlit page for uploading content in the AutoEval app.

Functionality:
- Upload a CSV file containing lesson plan data.
- Convert string data to JSON format.
- Insert the processed data into a PostgreSQL database.
"""
import json
import uuid

import pandas as pd
import streamlit as st

from utils.common_utils import log_message
from utils.formatting import convert_to_json
from utils.db_scripts import execute_multi_query

# System-generated fields to exclude when uploading JSON
SYSTEM_GENERATED_FIELDS = {
    '_id', 'id', 'index', 'deleted', 'duration', '_ts',
    'createdUtc', 'modifiedUtc', 'createdBy', 'modifiedBy',
    'parentId', 'docType', 'generatedBy', 'school', 'curriculum'
}


# Set page configuration
st.set_page_config(page_title="Upload Content", page_icon="⬆️")
st.markdown("# ⬆️ Upload Content")

st.header('CSV Upload and Process')
st.write("### Instructions for Uploading Data")

st.write(
    """
    1. **CSV File Format**: The CSV file should contain columns with 
    data to be converted to JSON format.
    2. **JSON Data**: If you have JSON data, ensure it is correctly
    formatted in the respective column.
    3. **String Data**: If the data is not in JSON format (e.g., plain
    text), it will be converted to a JSON object with the text stored 
    under the key `text`.
    """
)

st.write("### Example")

st.write("**JSON Data**:")
st.code(
    """
    {"name": "Lesson 1", "content": "This is a lesson plan."}
    """,
    language='json'
)

st.write("**String Data**:")
st.code("This is a plain text lesson plan.", language='text')

st.write("After conversion, it will be stored as:")
st.code(
    """
    {"text": "This is a plain text lesson plan."}
    """,
    language='json'
)

st.write("### How Your Data Will Be Processed")

st.write(
    """
    1. **Upload the CSV File**: Use the file uploader below to select 
    your CSV file.
    2. **Select the Column**: Choose the column that contains JSON data 
    or data to be converted to JSON.
    3. **Generation Details**: Provide a unique identifier or 
    description for your dataset. This will allow you to create a 
    dataset from the uploads. 
    4. **Insert into Database**: The data will be inserted into the 
    lesson plans table with each entry having a unique ID and the 
    provided generation details.
    """
)

st.write(
    "**Note**: Rows with missing (NaN) values in the selected column " 
    "will be skipped during the conversion process."
)
st.write(
    "Alternatively you can insert your data into the lesson plans "
    "table manually using SQL."
)
# File upload section with tabs for CSV and JSON
upload_tab1, upload_tab2 = st.tabs(["CSV Upload", "JSON Upload"])

with upload_tab1:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="csv_upload")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

        column_to_convert = st.selectbox(
            "Select the column which contains content in JSON format or to "
            "be converted to JSON",
            data.columns
        )

        data[column_to_convert] = data[column_to_convert].astype(str)
        data['json'] = data[column_to_convert].apply(convert_to_json)
        data = data[data['json'].notna()]

        st.write('Enter generation details for all rows.')
        generation_details = st.text_input(
            "This will help you differentiate your data from other entries "
            "in the lesson plans table when creating a dataset.",
            key="csv_gen_details"
        )

        data['id'] = [str(uuid.uuid4()) for _ in range(len(data))]
        data['generation_details'] = generation_details

        if st.button('Insert Data into Database', key="csv_insert"):
            queries_and_params = []
            for row in data.itertuples(index=False):
                query = """
                    INSERT INTO public.lesson_plans (id, json, generation_details)
                    VALUES (%s, %s, %s);
                """
                params = (row.id, json.dumps(row.json), row.generation_details)
                queries_and_params.append((query, params))

            result_message = execute_multi_query(queries_and_params)

            if result_message:
                log_message("success", "Data inserted successfully!")

with upload_tab2:
    st.write("### JSON File Upload")
    st.write(
        """
        Upload a JSON file containing one or more lesson plans.
        The file can contain either:
        - A single lesson plan object
        - An array of lesson plan objects
        """
    )

    uploaded_json = st.file_uploader(
        "Upload your JSON file",
        type=["json"],
        key="json_upload"
    )

    if uploaded_json is not None:
        try:
            # Read and parse JSON file
            json_content = json.load(uploaded_json)

            # Handle both single object and array of objects
            if isinstance(json_content, dict):
                json_records = [json_content]
            elif isinstance(json_content, list):
                json_records = json_content
            else:
                st.error("Invalid JSON format. Expected an object or array of objects.")
                json_records = []

            if json_records:
                st.success(f"Successfully loaded {len(json_records)} lesson plan(s)")

                # Display preview of the data
                st.write("### Preview of uploaded data:")
                for idx, record in enumerate(json_records[:3]):  # Show first 3 records
                    with st.expander(f"Record {idx + 1}"):
                        st.json(record)

                if len(json_records) > 3:
                    st.info(f"Showing first 3 of {len(json_records)} records")

                # Display extracted metadata
                st.write("### Extracted Metadata Preview")
                metadata_preview = []

                # Level to Key Stage conversion mapping (for preview)
                level_to_keystage_preview = {
                    'kindergarten': 'Key Stage 1',
                    'pre-primary': 'Key Stage 1',
                    'year 1': 'Key Stage 1',
                    'year 2': 'Key Stage 1',
                    'year 3': 'Key Stage 2',
                    'year 4': 'Key Stage 2',
                    'year 5': 'Key Stage 2',
                    'year 6': 'Key Stage 2',
                    'year 7': 'Key Stage 3',
                    'year 8': 'Key Stage 3',
                    'year 9': 'Key Stage 3',
                    'year 10': 'Key Stage 4',
                    'year 11': 'Key Stage 5',
                    'year 12': 'Key Stage 5'
                }

                for idx, record in enumerate(json_records[:3]):
                    subject = record.get('subject') or record.get('Subject') or 'Not found'
                    key_stage = record.get('keyStage') or record.get('key_stage') or record.get('KeyStage')

                    # Check for level conversion
                    converted_from_level = False
                    if not key_stage:
                        level = record.get('level') or record.get('Level')
                        if level:
                            level_normalized = str(level).lower().strip()
                            key_stage = level_to_keystage_preview.get(level_normalized, 'Unknown level')
                            converted_from_level = True if key_stage != 'Unknown level' else False
                        else:
                            key_stage = 'Not found'

                    metadata_preview.append({
                        'Record': idx + 1,
                        'Subject': subject,
                        'Key Stage': key_stage,
                        'Source': 'Converted from level' if converted_from_level else 'Direct from JSON'
                    })

                preview_df = pd.DataFrame(metadata_preview)
                st.dataframe(preview_df, hide_index=True, use_container_width=True)

                if len(json_records) > 3:
                    st.info(f"Preview shows metadata for first 3 of {len(json_records)} records")

                # Generation details input
                st.write('### Enter generation details for all records')
                json_generation_details = st.text_input(
                    "This will help you differentiate your data from other entries "
                    "in the lesson plans table when creating a dataset.",
                    key="json_gen_details",
                    value=""
                )

                # Insert button
                if st.button('Insert JSON Data into Database', key="json_insert"):
                    queries_and_params = []

                    # Level to Key Stage conversion mapping
                    level_to_keystage = {
                        'kindergarten': 'Key Stage 1',
                        'pre-primary': 'Key Stage 1',
                        'year 1': 'Key Stage 1',
                        'year 2': 'Key Stage 1',
                        'year 3': 'Key Stage 2',
                        'year 4': 'Key Stage 2',
                        'year 5': 'Key Stage 2',
                        'year 6': 'Key Stage 2',
                        'year 7': 'Key Stage 3',
                        'year 8': 'Key Stage 3',
                        'year 9': 'Key Stage 3',
                        'year 10': 'Key Stage 4',
                        'year 11': 'Key Stage 5',
                        'year 12': 'Key Stage 5'
                    }

                    for record in json_records:
                        record_id = str(uuid.uuid4())

                        # Clean the record by removing system-generated fields
                        cleaned_record = {
                            k: v for k, v in record.items()
                            if k not in SYSTEM_GENERATED_FIELDS
                        }

                        # Log removed fields for transparency
                        removed_fields = [k for k in record.keys() if k in SYSTEM_GENERATED_FIELDS]
                        if removed_fields:
                            log_message("info", f"Removed system fields: {', '.join(removed_fields)}")

                        # Extract subject from JSON
                        subject = record.get('subject') or record.get('Subject') or None

                        # Extract or convert key_stage
                        key_stage = record.get('keyStage') or record.get('key_stage') or record.get('KeyStage')

                        # If no keyStage found, try to convert from level
                        if not key_stage:
                            level = record.get('level') or record.get('Level')
                            if level:
                                # Normalize level string for matching
                                level_normalized = str(level).lower().strip()
                                key_stage = level_to_keystage.get(level_normalized)

                                if key_stage:
                                    log_message("info", f"Converted level '{level}' to '{key_stage}'")

                        query = """
                            INSERT INTO public.lesson_plans (id, json, generation_details, subject, key_stage)
                            VALUES (%s, %s, %s, %s, %s);
                        """
                        params = (record_id, json.dumps(cleaned_record), json_generation_details, subject, key_stage)
                        queries_and_params.append((query, params))

                    result_message = execute_multi_query(queries_and_params)

                    if result_message:
                        log_message("success", f"Successfully inserted {len(json_records)} lesson plan(s)!")
                        st.info(
                            f"You can now create a dataset using generation details: "
                            f"'{json_generation_details}' in the Build Datasets page."
                        )

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON file: {str(e)}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
