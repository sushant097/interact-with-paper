# Interact with Research Paper 

This project is an UI made it with Streamlit and apply embeddings models from OpenAI to read papers and ask them something related to the paper content! Explore more on the [Notebooks](./notebooks).

**This streamlit app is ready to deploy.**

## Features:
1. Paper Upload either Pdf or Arxiv paper link
2. Summarize the paper with OpenAI model. You can also use hugging face model for summarization. See this [notebook](notebooks/articlesummarizerhuggingface.ipynb).
3. Interact with paper as asking question. This works as follows:
   1. Get the chunks of text and calculate embeddings of each. 
      1. **In Inference:**
         1. New Question -> Calculate embedding 
         2. Get topK similar chunks through cosine similarity.

    

## Usage 🔩
1. Install requirements
```
pip install -r requirements.txt
```
2. Create a .env file with the apikey from OPENAI, with the following content
```
OPENAI_API_KEY
```

3. Run the app
```
streamlit run main.py
```

### Demo:
![](files/InteractWithPaperDemo_final.gif)


### References
[@keerthanpg](https://github.com/keerthanpg/talktopapers)

