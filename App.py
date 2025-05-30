import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
import requests
import re
import bs4 as bs
import math
import urllib.request
from nltk.stem import WordNetLemmatizer 
import spacy
nltk.download('wordnet')

#Initializing few variable
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer() 
nltk.download('punkt')

st.set_page_config(page_title='NewsSnap: AI Powered News Summarization📰 System', page_icon='./Meta/newspaper.ico')

def fetch_top_news():
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey=30afe5d6b1f545c6a9904eed97c6aeed' 
    try: 
        response = requests.get(url) 
        response.raise_for_status() # Raise an exception for HTTP errors 
        data = response.json() 
        if data['status'] == 'ok' and len(data['articles']) > 0: 
            return data['articles'] 
        else: 
            print("No articles found.") 
            return [] 
    except requests.exceptions.RequestException as e: 
        print(f"Error fetching news: {e}") 
        return []

def fetch_category_news(topic):
    url = f'https://newsapi.org/v2/top-headlines?country=us&category={topic}&apiKey=30afe5d6b1f545c6a9904eed97c6aeed' 
    try: 
        response = requests.get(url) 
        response.raise_for_status() # Raise an exception for HTTP errors 
        data = response.json() 
        print(len(data['articles']));
        if data['status'] == 'ok' and len(data['articles']) > 0: 
            return data['articles'] 
        else: 
            print("No articles found for the specified category.") 
            return [] 
    except requests.exceptions.RequestException as e: 
        print(f"Error fetching news: {e}") 
        return []

def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_container_width=True)
    except:
        image = Image.open('./Meta/no_image.jpg')
        st.image(image, use_container_width=True)


def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        news_data = Article(news['url'])
        text = ""
        try:
            scrap_data = urllib.request.urlopen(news['url'])
            article = scrap_data.read()
            parsed_article = bs.BeautifulSoup(article,'lxml')
            
            paragraphs = parsed_article.find_all('p')
            article_text = ""
            
            for p in paragraphs:
                article_text += p.text
            
            #Removing all unwanted characters
            article_text = re.sub(r'\[[0-9]*\]', '', article_text)
            text = genrate_summary(article_text)
        except Exception as e:
            pass
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            pass
        if(text != ""):
            c+=1
            st.write('**({}) {}**'.format(c, news['title'] ))
            
            fetch_news_poster(news_data.top_image)
            with st.expander(news['title']):
                st.markdown(
                    '''<h6 style='text-align: justify;'>{}"</h6>'''.format(text),
                    unsafe_allow_html=True)
                st.markdown("[Read more at ...{}]".format(news['url']))
            st.success("Published Date: " + news['publishedAt'])
        if c >= news_quantity:
                break











def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words

    for sent in sentences:
        freq_table = {} #dictionary with 'words' as key and their 'frequency' as value
        
        #Getting all word from the sentence in lower case
        words = [word.text.lower() for word in sent  if word.text.isalnum()]
       
        for word in words:  
            word = lemmatizer.lemmatize(word)   #Lemmatize the word
            if word not in stopWords:           #Reject stopwords
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

        freq_matrix[sent[:15]] = freq_table

    return freq_matrix


#Function to calculate Term Frequency(TF) of each word
#INPUT -> freq_matrix
#OUTPUT -> tf_matrix (A dictionary with each sentence itself as key, 
# and a dictionary of words of that sentence with their Term-Frequency as value)

#TF(t) = (Number of times term t appears in  document) / (Total number of terms in the document)
def tfMatrix(freq_matrix):
    tf_matrix = {}

    for sent, freq_table in freq_matrix.items():
        tf_table = {}  #dictionary with 'word' itself as a key and its TF as value

        total_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


#Function to find how many sentences contain a 'word'
#INPUT -> freq_matrix
#OUTPUT -> sent_per_words (Dictionary with each word itself as key and number of 
#sentences containing that word as value)

def sentences_per_words(freq_matrix):
    sent_per_words = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in sent_per_words:
                sent_per_words[word] += 1
            else:
                sent_per_words[word] = 1

    return sent_per_words


#Function to calculate Inverse Document frequency(IDF) for each word
#INPUT -> freq_matrix,sent_per_words, total_sentences
#OUTPUT -> idf_matrix (A dictionary with each sentence itself as key, 
# and a dictionary of words of that sentence with their IDF as value)

#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
def idfMatrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


#Function to calculate Tf-Idf score of each word
#INPUT -> tf_matrix, idf_matrix
#OUTPUT - > tf_idf_matrix (A dictionary with each sentence itself as key, 
# and a dictionary of words of that sentence with their Tf-Idf as value)
def tf_idfMatrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

       #word1 and word2 are same
        for (word1, tf_value), (word2, idf_value) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(tf_value * idf_value)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


#Function to rate every sentence with some score calculated on basis of Tf-Idf
#INPUT -> tf_idf_matrix
#OUTPUT - > sentenceScore (Dictionary with each sentence itself as key and its score
# as value)
def score_sentences(tf_idf_matrix):
    
    sentenceScore = {}

    for sent, f_table in tf_idf_matrix.items():
        total_tfidf_score_per_sentence = 0

        total_words_in_sentence = len(f_table)
        for word, tf_idf_score in f_table.items():
            total_tfidf_score_per_sentence += tf_idf_score

        if total_words_in_sentence != 0:
            sentenceScore[sent] = total_tfidf_score_per_sentence / total_words_in_sentence

    return sentenceScore



#Function Calculating average sentence score 
#INPUT -> sentence_score
#OUTPUT -> average_sent_score(An average of the sentence_score) 
def average_score(sentence_score):
    
    total_score = 0
    for sent in sentence_score:
        total_score += sentence_score[sent]

    average_sent_score = (total_score / len(sentence_score))

    return average_sent_score


#Function to return summary of article
#INPUT -> sentences(list of all sentences in article), sentence_score, threshold
# (set to the average pf sentence_score)
#OUTPUT -> summary (String text)
def create_summary(sentences, sentence_score, threshold):
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= (threshold):
            summary += " " + sentence.text
        

    return summary

def genrate_summary(text) :
    original_words = text.split()
    original_words = [w for w in original_words if w.isalnum()]
    num_words_in_original_text = len(original_words)

    #Converting received text into sapcy Doc object
    text = nlp(text)

    #Extracting all sentences from the text in a list
    sentences = list(text.sents)
    total_sentences = len(sentences)

    #Generating Frequency Matrix
    freq_matrix = frequency_matrix(sentences)

    #Generating Term Frequency Matrix
    tf_matrix = tfMatrix(freq_matrix)

    #Getting number of sentences containing a particular word
    num_sent_per_words = sentences_per_words(freq_matrix)

    #Generating ID Frequency Matrix
    idf_matrix = idfMatrix(freq_matrix, num_sent_per_words, total_sentences)

    #Generating Tf-Idf Matrix
    tf_idf_matrix = tf_idfMatrix(tf_matrix, idf_matrix)


    #Generating Sentence score for each sentence
    sentence_scores = score_sentences(tf_idf_matrix)

    #Setting threshold to average value (You are free to play with ther values) 
    threshold = average_score(sentence_scores)

    #Getting summary 
    summary = create_summary(sentences, sentence_scores, 1.3 * threshold)
    # print("\n\n")
    # print("*"*20,"Summary","*"*20)
    # print("\n")
    # print(summary)
    # print("\n\n")
    # print("Total words in original article = ", num_words_in_original_text)
    # print("Total words in summarized article = ", len(summary.split()))
    return summary
    





def run():
    st.title("NewsSnap: AI Powered News Summarization 📰 System")
    image = Image.open('./Meta/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_container_width=False)

    with col3:
        st.write("")
    category = ['--Select--', 'Trending🔥 News', 'Favourite💙 Topics', 'Search🔍 Topic']
    cat_op = st.selectbox('Select your Category', category)
    if cat_op == category[0]:
        st.warning('Please select Type!!')
    elif cat_op == category[1]:
        st.subheader("✅ Here is the Trending🔥 news for you")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        news_list = fetch_top_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        av_topics = ['Choose Topic', 'GENERAL', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE',
                     'HEALTH']
        st.subheader("Choose your favourite Topic")
        chosen_topic = st.selectbox("Choose your favourite Topic", av_topics)
        if chosen_topic == av_topics[0]:
            st.warning("Please Choose the Topic")
        else:
            no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
            news_list = fetch_category_news(chosen_topic)
            if news_list:
                st.subheader("✅ Here are the some {} News for you".format(chosen_topic))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(chosen_topic))
    elif cat_op == category[3]:
        user_topic = st.text_input("Enter your Topic🔍")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=15, step=1)

        if st.button("Search") and user_topic != '':
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_category_news(topic=user_topic_pr)
            if news_list:
                st.subheader("✅ Here are the some {} News for you".format(user_topic.capitalize()))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(user_topic))
        else:
            st.warning("Please write Topic  to Search🔍")

run()