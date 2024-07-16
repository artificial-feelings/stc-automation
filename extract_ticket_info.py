import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai
import pandas as pd
import numpy as np
import os
import json
import base64
from datetime import datetime

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser



class TicketDataExtraction(BaseModel):
    userFullName: str = Field(description="First and Last Name")
    flightDate: datetime = Field(description="Date of the flight")
    flightNumber: str = Field(description="Flight Number")
    cityDeparture: str = Field(description="City or airport code of departure")
    cityDestination: str = Field(description="City or airport code of destination")
    ticketClass: str = Field(description="Ticket class. Usually business or economy.")

class Tickets(BaseModel):
    tickets: List[TicketDataExtraction]


def load_google_credentials():
    credentials_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
    if not credentials_base64:
        raise ValueError("No credentials found in environment variables.")
    
    credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
    credentials_info = json.loads(credentials_json)
    
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    return credentials


def build_sheets_service(credentials):
    service = build('sheets', 'v4', credentials=credentials)
    return service.spreadsheets()

def read_google_sheet(sheet_service, sheet_id):
    result = sheet_service.values().get(spreadsheetId=sheet_id, range="main").execute()
    values = result.get('values', [])

    columns = values[0]
    data = values[1:]

    for row in data:
        while len(row) < len(columns):
            row.append('')

    df = pd.DataFrame(values[1:], columns=values[0]) if values else pd.DataFrame()
    return df

def append_to_dataframe(df: pd.DataFrame, new_tickets: Tickets, filename: str) -> pd.DataFrame:
    if not new_tickets.tickets:  # Check if the results are empty
        new_data = [{col: '' for col in df.columns}]
        new_data[0]['FileName'] = filename
    else:
        new_data = [ticket.dict() for ticket in new_tickets.tickets]
        for entry in new_data:
            entry['FileName'] = filename
            entry['Full Name'] = entry.pop('userFullName', '')
            entry['Flight'] = entry.pop('flightNumber', '')
            entry['Date'] = entry.pop('flightDate', '')
            entry['City of Departure'] = entry.pop('cityDeparture', '')
            entry['City of Arrival'] = entry.pop('cityDestination', '')
            entry['Class'] = entry.pop('ticketClass', '')
            entry['Price'] = None  # Assuming 'Price' is not provided and needs to be added

            if entry['Date'] != "":
                entry['Date'] = entry['Date'].strftime("%Y-%m-%d %H:%M")

    new_df = pd.DataFrame(new_data)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    
    return updated_df

def update_google_sheet(sheet_service, sheet_id, df):
    values = df.values.tolist()
    values.insert(0, df.columns.tolist())
    sheet_service.values().update(
        spreadsheetId=sheet_id,
        range="main",
        valueInputOption="RAW",
        body={"values": values}
    ).execute()

@st.cache_resource
def load_model():
    return ChatOpenAI(model_name="gpt-4o", temperature=0)


def process_file(file_name: str):
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()
    text = " ".join(list(map(lambda page: page.page_content, pages)))

    parser = PydanticOutputParser(pydantic_object=Tickets)
    prompt = PromptTemplate(
        template="Extract necessary information from document.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = load_model()
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": text})
    results = parser.invoke(output)
    return results

def get_ticket_wording(df):
    last_symbol = str(len(df))[-1]
    if last_symbol == "1":
        ending = ""
    elif last_symbol in ["2", "3", "4"]:
        ending = "а"
    else:
        ending = "ов"
    return ending


def extract_ticket_info_layout():
    st.title("Внести билеты в таблицу")
    st.write("Загрузите билеты в формате PDF для извлечения информации и загрузки в Google Sheets.")

    if 'file_uploader_key' not in st.session_state:
        st.session_state['file_uploader_key'] = 0


    uploaded_files = st.file_uploader("Choose PDF files",
                                      type="pdf",
                                      accept_multiple_files=True,
                                      key=st.session_state["file_uploader_key"])
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_sheet_id = os.getenv("GOOGLE_SHEET_ID")
    
    if openai_api_key and google_sheet_id:
        credentials = load_google_credentials()
        sheet_service = build_sheets_service(credentials)
        df = read_google_sheet(sheet_service, google_sheet_id)
        new_df = pd.DataFrame(columns=list(df.columns))

        if uploaded_files:

            col1, col2 = st.columns([5, 2])
            with col1:
                process_button = st.button("Обработать PDF")
            with col2:
                clear_button = st.button("Очистить список")

            if clear_button:
                st.session_state['file_uploader_key'] += 1
                st.experimental_rerun()

            if process_button:
                # Initialize the progress bar
                progress = st.progress(0)
                total_files = len(uploaded_files)
                processed_files = 0
                error_count = 0
                
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        results = process_file(uploaded_file.name)
                        new_df = append_to_dataframe(new_df, results, uploaded_file.name)
                        processed_files += 1
                    except Exception as e:
                        st.error(f"Ошибка при обработке файла {uploaded_file.name}: {e}")
                        error_count += 1
                    
                    # Update the progress bar
                    progress.progress((i + 1) / total_files)
                    status_text.text(f"Обработано {processed_files} из {total_files} файлов. Ошибки: {error_count}")

                df = read_google_sheet(sheet_service, google_sheet_id)
                df = pd.concat([df, new_df], ignore_index=True)
                if 'Price' in df.columns:
                    df['Price'] = df['Price'].apply(lambda x: np.nan if (x is None) or (x == "") else x)
                    df['Price'] = df['Price'].astype(float)
                update_google_sheet(sheet_service, google_sheet_id, df)
                
                st.write("Обработка билетов завершена")
                
                # Clear the list of uploaded files
                st.session_state['file_uploader_key'] += 1
                st.rerun()
        
        # Display the current content of the Google Sheet
        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        with col1:
            ending = get_ticket_wording(df)
            st.write(f"На данный момент в таблице {len(df)} билет{ending}:")

        with col2:
            if st.button("Обновить"):
                df = read_google_sheet(sheet_service, google_sheet_id)
                st.rerun()
        st.dataframe(df)
        
        sheet_url = f"https://docs.google.com/spreadsheets/d/{google_sheet_id}/edit"
        st.markdown(f"[Перейти в таблицу Google Sheet]({sheet_url})")
    else:
        st.error("Please set the OPENAI_API_KEY and GOOGLE_SHEET_ID environment variables.")
