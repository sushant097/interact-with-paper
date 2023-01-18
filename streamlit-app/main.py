from utils import parse_paper, check_folders, download_arxiv
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

@st.cache(allow_output_mutation=True)
def process_file(file):
    print('[INFO] Processing and calc embeddings')
    # Process the file and filter
    with st.spinner(text='Procesing your paper'):
        paper_text = parse_paper(file)
        # Apply some filter
        filtered_paper_text = []
        for row in paper_text:
            if len(row['text']) < 30:
                continue
            filtered_paper_text.append(row)
        paper_df = pd.DataFrame(filtered_paper_text)
        
    # Calculate embeddings
    
    with st.spinner(text='Calculate embeddings'):
        embedding_model = "text-embedding-ada-002"
        embeddings = paper_df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        paper_df["embeddings"] = embeddings
    return paper_df
        
# Check if data folder exists
check_folders()

if __name__ == '__main__':
    
    st.title('Ask the Paper 📚')
    
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

            paper = process_file(f'data/pdf/{uploaded_file.name}')
            
            st.subheader('Now You are ready to ask to your paper')
            
            text_input = st.text_input(
            "Ask your paper here and hit enter 👇",
            label_visibility='visible',
            placeholder='Your Question',
            )
            if text_input:
                print(text_input)
                results = search(paper, text_input, n=3)
                st.write(results.iloc[0]['text'])
                

    else:
        text_input = st.sidebar.text_input(
            "Enter your arxiv Link here 👇",
            label_visibility='visible',
            placeholder='Arvix Link',
        )

        if text_input:
            st.sidebar.write("You entered: ", text_input)
            with st.spinner(text='Procesing your link...'):
                download_arxiv(text_input)
                
            paper = process_file('data/pdf/downloaded-paper.pdf')
            
            st.subheader('Now You are ready to ask to your paper')
            
            text_input = st.text_input(
            "Ask your paper here and hit enter 👇",
            label_visibility='visible',
            placeholder='Your Question',
            )
            if text_input:
                print(text_input)
                results = search(paper, text_input, n=3)
                st.write(results.iloc[0]['text'])
