import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

# set streamlit application
st.set_page_config(page_title="Summarize Text From YT or Website", page_icon="🦜")

st.title("🦜 LangChain: Summarize Text From YT or Website")

st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    GROQ_API_KEY=st.text_input(label="Groq API key",type="password",value="")

generic_url=st.text_input(label="URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm = ChatGroq(api_key=GROQ_API_KEY,model="Gemma-7b-It")

#prompt template
prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=['text'])

if st.button(label="Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not GROQ_API_KEY.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video url or website url")
    else:
        try:
            with st.spinner("wait for it..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")