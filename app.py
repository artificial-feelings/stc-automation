import fitz # PyMuPDF
import streamlit as st

import json
import zipfile
from io import BytesIO


def check_credentials(username, password):
    if username in st.secrets.users and st.secrets.users[username] == password:
        return True
    return False


def login():
    username = st.text_input("Имя пользователя", key="username_input")
    password = st.text_input("Пароль", type="password", key="password_input")
    if st.button("Войти"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Неверное имя пользователя или пароль")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""


def overlay_area_between_titles(pdf_file, areas):
    pdf_bytes = pdf_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for area in areas:
      for page_num in range(len(pdf_document)):
          page = pdf_document[page_num]
          text_instances1 = page.search_for(area["from"])
          text_instances2 = page.search_for(area["to"])
          
          if text_instances1 and text_instances2:
              rect1 = text_instances1[0] 
              rect2 = text_instances2[0]
            
              area_to_remove = fitz.Rect(int(rect1.x0+area["left_offset"]), int(rect1.y0+area["top_offset"]), page.rect.width, int(rect2.y0+area["bottom_offset"]))
              page.draw_rect(area_to_remove, color=(1, 1, 1), fill=(1, 1, 1))
    output_pdf = BytesIO()
    pdf_document.save(output_pdf)
    output_pdf.seek(0)
    
    return output_pdf

def pdf_processor():

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    st.title("Обработка билетов PDF")
    with open("templates.json", "rb") as f:
        templates = json.load(f)
    
    uploaded_files = st.file_uploader("Билеты в формате PDF",
                                      type="pdf",
                                      accept_multiple_files=True,
                                      key=st.session_state["file_uploader_key"])

    if uploaded_files:
        if st.button("Очистить список"):
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()

    current_template = st.selectbox("Шаблон билета", list(templates.keys()))

    if st.button("Обработать PDF"):
        if uploaded_files:
            zip_buffer = BytesIO()
    
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for uploaded_file in uploaded_files:
                    # Process each uploaded PDF
                    processed_pdf = overlay_area_between_titles(uploaded_file, templates[current_template])
    
                    # Add processed PDF to the zip file
                    zip_file.writestr("processed_" + uploaded_file.name, processed_pdf.read())
    
            zip_buffer.seek(0)
    
            # Provide download button for the zip file
            st.download_button(
                label="Скачать PDF",
                data=zip_buffer,
                file_name="processed_tickets.zip",
                mime="application/zip"
            )
        else:
            st.error("Сначала загрузите PDF билеты.")


def other_functionality():
    st.title("Other Functionality")
    st.write("This is another functionality screen.")


def render_main():
    st.sidebar.title("")
    st.sidebar.button("Выйти", on_click=logout)
    options = st.sidebar.selectbox("Функционал", ["Обработка билетов PDF", "Other Functionality"])
    if options == "Обработка билетов PDF":
        pdf_processor()
    elif options == "Other Functionality":
        other_functionality()


if __name__=="__main__":

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state["logged_in"]:
        render_main()
    else:
        login()

