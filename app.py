from io import StringIO
import pandas as pd
import streamlit as st
from engine import analyze
st.set_page_config(page_title='Sentiment Explorer | Mayimunah Nagayi',page_icon='💬',layout='wide')
st.markdown('''<style>.block-container{max-width:1150px;padding-top:3rem}h1{font-family:Georgia,serif}.stMetric{background:#eee8ec;padding:20px;border-radius:12px}</style>''',unsafe_allow_html=True)
st.caption('MAYIMUNAH NAGAYI · HONOURS RESEARCH')
st.title('Exploring patterns in public sentiment')
st.write('An interactive reconstruction of my Honours project on unsupervised tweet analysis using affinity propagation, with K-means as a baseline.')
explore,method=st.tabs(['Explore texts','About the project'])
with explore:
    st.info('This recovered-model demo shows cluster assignments. Cluster numbers are not verified positive, neutral or negative sentiment labels.')
    mode=st.radio('Choose your input',['Paste text','Upload CSV'],horizontal=True)
    texts=[]
    if mode=='Paste text':
        raw=st.text_area('One text per line',value='Just saw the most amazing movie! Highly recommend it.\nThis phone is such a disappointment.\nPlanning a trip to Europe soon.\nFeeling grateful for my friends and family.',height=170,max_chars=100000)
        texts=[t.strip() for t in raw.splitlines() if t.strip()]
    else:
        f=st.file_uploader('Upload a CSV with a text column',type=['csv'])
        if f:
            try:
                df=pd.read_csv(f)
                col=st.selectbox('Text column',df.columns)
                texts=[str(t).strip() for t in df[col].dropna() if str(t).strip()]
                st.caption(f'{len(texts):,} nonempty texts found')
            except Exception as e:st.error(f'Could not read this CSV: {e}')
    if st.button('Analyse texts',type='primary'):
        if not texts:st.warning('Add at least one text first.')
        elif len(texts)>500:st.warning('Please use up to 500 texts per analysis.')
        else:
            with st.spinner('Comparing texts with the saved cluster centres…'):
                result=analyze(texts)
            a,b,c=st.columns(3)
            a.metric('Texts',len(result));b.metric('AP clusters represented',result.loc[result.Status=='Assigned','AP cluster'].nunique());c.metric('K-means clusters represented',result.loc[result.Status=='Assigned','K-means cluster'].nunique())
            st.subheader('Your results')
            st.dataframe(result,hide_index=True,use_container_width=True)
            st.download_button('Download results',result.to_csv(index=False).encode(),file_name='cluster_results.csv',mime='text/csv')
            st.subheader('K-means cluster distribution')
            st.bar_chart(result['K-means cluster'].value_counts())
with method:
    st.subheader('Unsupervised Sentiment Analysis using Affinity Propagation')
    st.write('Mayimunah Nagayi · University of the Western Cape · Supervisor Prof Clement N. Nyirenda')
    st.write('The original project explored tweet clustering using TF-IDF features, affinity propagation and agglomerative clustering, with K-means for comparison.')
    st.markdown('''**What this reconstruction does**

It reuses the saved TF-IDF vocabulary and weights, SVD components, affinity propagation exemplars and K-means centres from the original repository. New texts are assigned to their nearest saved centre. No model is trained during use.

**Interpretation and limits**

The original testing notebooks include batch refitting and inconsistent prediction paths. This version uses fixed saved centres for repeatable inference. It does not reproduce batch refitting or claim the original evaluation scores. Agglomerative sentiment mapping remains unverified. Numeric clusters can group topic or wording rather than sentiment. English-only cleaning removes punctuation, numbers and stopwords, including some negation words; this limits sentiment interpretation.

The saved AP model represents the fitted state available in the repository, not a verified model of the entire training corpus. This reconstruction is a research demonstration, not a validated sentiment classifier.''')
    st.link_button('Original code','https://github.com/Mayimunah/Affinity-progation-sentiment-analysis')
    st.link_button('Project website and final demo','https://sites.google.com/myuwc.ac.za/unsupervisedsentimentanalysis')
st.divider()
st.caption('Honours research · Reconstructed interactive demonstration')
