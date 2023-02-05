from utils import parse_paper, parse_paper2, check_folders, download_arxiv
from openai.embeddings_utils import get_embedding, cosine_similarity
import streamlit as st
import pandas as pd
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def search(df, query, n=3, pprint=True):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
    )
    return results


def generate_summary(text):
    prompt = "summarize in short: " + text + "\n Tl;dr:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=120,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    return response["choices"][0]["text"]


@st.cache(allow_output_mutation=True)
def process_file2(file):
    print('[INFO] Processing and calc embeddings')
    # Process the file and filter
    with st.spinner(text='Procesing your paper and generating summary...'):
        doc = parse_paper2(file)
        summary_list = []
        for page in doc:
            text = page.get_text("text")
            summary_list.append(generate_summary(text))
        paper_summary1 = '<br>'.join(summary_list)
        paper_summary2 = ''.join(summary_list)

        summary_list = paper_summary2.split('. ')
        for i, sentence in enumerate(summary_list):
            summary_list[i] = sentence + '. '
        paper_df = pd.DataFrame(summary_list)

    with st.spinner(text='Calculate embeddings'):
        embedding_model = "text-embedding-ada-002"
        embeddings = paper_df[0].astype(str).apply([lambda x: get_embedding(x, engine=embedding_model)])
        paper_df["embeddings"] = embeddings

    return len(doc), paper_summary1, paper_df


# Check if data folder exists else create it
check_folders()

if __name__ == '__main__':

    st.title('Interact with Paper 📚')

    source = ("PDF", "ARXIV LINK")
    source_index = st.sidebar.selectbox("Select Input type", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Load File", type=['pdf'])
        if uploaded_file is not None:
            # Upload pdf file
            with st.spinner(text='Uploading your pdf...'):
                with open(f'data/pdf/{uploaded_file.name}', mode='wb') as w:
                    w.write(uploaded_file.getvalue())

            total_pages, paper_summary, paper_df = process_file2(f'data/pdf/{uploaded_file.name}')
            # paper_df = process_file(f'data/pdf/{uploaded_file.name}')

            st.write(f"<b>Paper Summary of Total Pages: {total_pages} </b><br>", unsafe_allow_html=True)
            st.write(paper_summary, unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)

            st.subheader('Now You are ready to ask to your paper')

            text_input = st.text_input(
                "Ask your paper here and hit enter 👇",
                # label_visibility='visible',
                placeholder='Your Question',
            )
            if text_input:
                print(text_input)
                results = search(paper_df, text_input, n=3)
                print(results.iloc[0][0])
                st.write(results.iloc[0][0])


    else:
        text_input = st.sidebar.text_input(
            "Enter your arxiv Link here 👇",
            # label_visibility='visible',
            placeholder='Arvix Link',
        )

        if text_input:
            st.sidebar.write("You entered: ", text_input)
            with st.spinner(text='Procesing your link...'):
                download_arxiv(text_input)

            total_pages, paper_summary, paper_df = process_file2('data/pdf/downloaded-paper.pdf')

            st.write(f"<b>Paper Summary of Total Pages: {total_pages} </b><br>", unsafe_allow_html=True)
            st.write(paper_summary, unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)

            st.subheader('Now You are ready to ask to your paper')

            text_input = st.text_input(
                "Ask your paper here and hit enter 👇",
                # label_visibility='visible',
                placeholder='Your Question',
            )
            if text_input:
                print(text_input)
                results = search(paper_df, text_input, n=3)
                st.write(results.iloc[0][0]+'.')
