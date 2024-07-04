import fitz # PyMuPDF
import streamlit as st

import json
import zipfile
from io import BytesIO

from hide_prices import hide_prices_layout
from extract_ticket_info import extract_ticket_info_layout


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


def other_functionality():
    st.title("Other Functionality")
    st.write("This is another functionality screen.")


def render_main():
    st.sidebar.title("")
    st.sidebar.button("Выйти", on_click=logout)
    if st.session_state.username == "admin":
        functionality = ["Внести билеты PDF", "Обработать цены в PDF"]
    elif st.session_state.username == "stcuser":
        functionality = ["Внести билеты PDF"]
    else:
        functionality = []

    options = st.sidebar.selectbox("Функционал", functionality)

    if options == "Обработать цены в PDF":
        if st.session_state.username == "admin":
            hide_prices_layout()
    elif options == "Внести билеты PDF":
        extract_ticket_info_layout()


if __name__=="__main__":

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state["logged_in"]:
        render_main()
    else:
        login()

