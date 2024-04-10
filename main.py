import pandas as pd 
import streamlit as st
import re
from cleantext.sklearn import CleanTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


sw=list(ENGLISH_STOP_WORDS)
sw.remove("not")
def text_cleaning(doc):
    doc=re.sub("[^a-zA-Z ]","",doc)
    return doc
df=pd.read_csv("RestaurentReviwAnalysis\Restaurant_Reviews.tsv",sep="\t")
corpus=df.iloc[:,0].values
target=df.iloc[:,-1].values

Final_corpus=list(map(text_cleaning,corpus))

cv=CountVectorizer(lowercase=True,stop_words=sw)
X=cv.fit_transform(Final_corpus).toarray()
y=target
model=MultinomialNB()
model.fit(X,y)

st.header("Restaurant Sentiment Analysis",)
with st.expander("Analyze Text"):
    text=st.text_input("Text Here: ")
    if text:
        clean=text_cleaning(text)
        sample=cv.transform([clean]).toarray()
        prediction=model.predict(sample)
        if prediction==1:
            st.write("Positive :smile: :yum: :yum:")
        else:
            st.write("Negative 	:cry: 	:cry:	:pensive:")
    
    pre = st.text_input("Clean Text: ") 
    if pre:
        words=pre.split()
        newdoc=''
        for word in words:
            if word not in sw:
                newdoc=newdoc+word+" "
        cleaner = CleanTransformer(no_punct=True, lower=True,no_numbers=True,replace_with_number=" ",
                            no_urls=True,replace_with_url=" ",no_emails=True,replace_with_email=" ",no_digits=True,replace_with_digit=" ",no_emoji=True,no_currency_symbols=True,replace_with_currency_symbol=" ")
        cleantext=cleaner.transform([newdoc])
        st.write(cleantext[0])


with st.expander("Analyze CSV"):
    upl=st.file_uploader("Upload File")
    def score(pre):
        clean=text_cleaning(pre)
        sample=cv.transform([clean]).toarray()
        prediction=model.predict(sample)
        if prediction==1:
            return "Liked"
        else:
            return "Not liked"

    if upl:
        df=pd.read_csv(upl)
        # del df['Unnamed : 0']
        df['Analysis']=df['Review'].apply(score)
        st.write(df.head(10))
    
        @st.cache_data
        def convert_df(df):
            # IMPORTANT : Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
