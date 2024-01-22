

# To read the PDF
import PyPDF2
# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
# To extract text from tables in PDF
import pdfplumber
# To remove the additional created files
import os
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import gensim
from gensim import corpora
import pickle
import numpy as np

###################### SCRIPT 1: SET UP SHELL DATAFFRAME ######################
###############################################################################
# Global comment: path ends with /
directory_path = 'C:/Users/tnguyen4/Documents/BigO/'

paper_path = directory_path + 'papers/'
output_path = directory_path + 'output/'

file_list = os.listdir(paper_path)
df = pd.DataFrame({'filename': file_list})

# Create filepath variable
df['filepath'] = paper_path + df['filename']

# Extract the year from the filenames using the regular expression
df['publication_year'] = df['filename'].str.extract(r'(\d{4})')

# Initiate variables for page number and raw text
df['page_num'] = 0
df['raw_pdf_text'] = ""

###################### EXTRACT TEXT FROM PDFS #################################
###############################################################################

# Create a function to extract text
def text_extraction(element):
    # Extracting the text from the in-line text element
    line_text = element.get_text()
    
    # Find the formats of the text
    # Initialize the list with all the formats that appeared in the line of text
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            # Iterating through each character in the line of text
            for character in text_line:
                if isinstance(character, LTChar):
                    # Append the font name of the character
                    line_formats.append(character.fontname)
                    # Append the font size of the character
                    line_formats.append(character.size)
    # Find the unique font sizes and names in the line
    format_per_line = list(set(line_formats))
    
    # Return a tuple with the text in each line along with its format
    return (line_text, format_per_line)

# -------------------- Extract Text from PDF -------------------- 
for p in range(0,len(df),1):
# create a PDF file object
    pdf_path = df['filepath'][p]
    pdfFileObj = open(pdf_path, 'rb')
    # create a PDF reader object
    pdfReaded = PyPDF2.PdfReader(pdfFileObj)
    
    # Create the dictionary to extract text from each image
    text_per_page = {}
    
    # We extract the pages from the PDF
    for pagenum, page in enumerate(extract_pages(pdf_path)):
        
        # Initialize the variables needed for the text extraction from the page
        pageObj = pdfReaded.pages[pagenum]
        page_content = []
        # Initialize the number of the examined tables
        first_element= True
        table_extraction_flag= False
        # Open the pdf file
        pdf = pdfplumber.open(pdf_path)
        # Find the examined page
        page_tables = pdf.pages[pagenum]
        # Find the number of tables on the page
        tables = page_tables.find_tables()
    
        # Find all the elements
        page_elements = [(element.y1, element) for element in page._objs]
        
        # Sort all the elements as they appear in the page 
        page_elements.sort(key=lambda a: a[0], reverse=True)
    
        # Find the elements that composed a page
        for i,component in enumerate(page_elements):
            # Extract the position of the top side of the element in the PDF
            pos= component[0]
            # Extract the element of the page layout
            element = component[1]
            
            # Check if the element is a text element
            if isinstance(element, LTTextContainer):
                if table_extraction_flag == False:
                    # Use the function to extract the text and format for each text element
                    (line_text, format_per_line) = text_extraction(element)
                    # Append the text of each line to the page text
                    page_content.append(line_text)
                else: # Omit the text that appeared in a table/figure (other forms)
                    pass
    
            # Check the elements for images
            if isinstance(element, LTFigure):
                pass
    
            # Check the elements for tables
            if isinstance(element, LTRect):
                pass
    
        # Create the key of the dictionary
        dctkey = 'Page_'+str(pagenum)
        # Add the list of list as the value of the page key
        text_per_page[dctkey]= [page_content]
    
    # Closing the pdf file object
    pdfFileObj.close()
    
    text_per_pdf = ""
    
    # Collapse all pages together
    for page in text_per_page.keys():
        collapsed_text = ''.join(text_per_page[page][0])
        text_per_pdf = text_per_pdf + collapsed_text
    
    df.loc[p,'raw_pdf_text'] = text_per_pdf #Add text in the whole pdf 
    df.loc[p,'page_num'] = len(text_per_page.keys()) #Add pdf num of pages
    print("Doc",p, "finished") #see progress

with open(output_path + 'corpus_data_raw.pkl', 'wb') as file:
    pickle.dump(df, file)

    
################# SCRIPT 2: PROCESSING & CLEANING PDF TEXT ################
###########################################################################

#Processing steps:
    #(1) lower case, (2) remove linebreak and double spacing, (3) remove punctuations, (4) remove stopwords
    #(5) stemming/lemmatizing #choose lemmatizing (if want more aggressive, can use Porter stemming later)
    
punctuation = string.punctuation +'”' + '“' + '’'
stop = stopwords.words('english') 
lemma = WordNetLemmatizer()

def remove_stop_words(text):
    stopWordsRemovedDoc = " ".join([word for word in text.split() if word not in stop])
    return stopWordsRemovedDoc

def remove_punc(text):
    puncRemoved = "".join([char for char in text if char not in punctuation])
    return puncRemoved

def lemmatize_doc(text):
    docLemmatized = " ".join([lemma.lemmatize(word) for word in text.split()])
    return docLemmatized

to_keep = ['uk', '3d', 'eu', 'rd', '5g', 'pc', 'ex', '3g', '4g', '2g', '6g']

def remove_single_letter(text): #single or 2-letter
    singleLetterRemoved = " ".join([word for word in text.split() if (len(word) > 2) or (word in to_keep)])
    return singleLetterRemoved

df['clean_pdf_text'] = df['raw_pdf_text'].str.lower().replace('\n',' ')
df['clean_pdf_text'] = df['clean_pdf_text'].apply(remove_stop_words)
df['clean_pdf_text'] = df['clean_pdf_text'].apply(remove_punc).apply(remove_single_letter)
df['lem_pdf_text'] = df['clean_pdf_text'].apply(lemmatize_doc)

# Create list/bag of words for each paper
df['bag_of_word'] = df['lem_pdf_text'].apply(lambda x: x.split())

# Cleaning 2nd round - remove additional single-letter word
stop_2 = []
for i in range(0,len(df)): #loop through each papert
    tmp = [word for word in df.loc[i,'bag_of_word'] if len(word) <= 2]
    stop_2.extend(tmp) #57,906
stop_2 = set(stop_2) #816 - review this list to get to_keep list
# Cleaning 2nd round - end 

# Create 2 variables to store the number of words pre and post cleaning 2nd round 
df['num_bag_of_word'] = df['bag_of_word'].apply(lambda x: len(x))

# Export data as a pickle object:
if os.path.exists(output_path) == False:
    os.mkdir(output_path)
    
with open(output_path + 'corpus_data_processed.pkl', 'wb') as file:
    pickle.dump(df, file)

###################### TRAINING & APPLYING LDA MODEL #######################
#############################################################################

LDA = gensim.models.ldamodel.LdaModel

# Create a dictionary using bag of words cleaned
dictionary = corpora.Dictionary(df['bag_of_word'])

# Word count for each word per each research paper (word frequency matrix)
docTermMatrix = [dictionary.doc2bow(text) for text in df['bag_of_word']]
type(docTermMatrix) #list

# See what is the top 100 most popular words
termCumFreq = dictionary.token2id #access token2id attribute in dictionary class
Freqlist = list(termCumFreq.items())
# Freqlist.sort(key=lambda n: n[1], reverse = True)
# top100 = Freqlist[:100]

termCumFreq['ziguang']

#Y tuong: tao array tu tat ca cac tu trong dict[1] (nhu 1 column)
#loop qua doc term matrix, i = 1: document 1, loop through k in the tuple
#for each tuple in k tuple, tuple[k][0] = index 

index_array = np.array(range(0,len(termCumFreq),1))

# Create word frequency matrix for each document

key_word = [0]*len(Freqlist)

for i in range(0,len(Freqlist)):
    pair = Freqlist[i]
    key_word[pair[1]] = pair[0]
    
frequency_matrix = pd.DataFrame({'Index': index_array})
frequency_matrix['word'] = key_word

for i in range(0,len(docTermMatrix)): #loop through each doc
    freq = [0]*len(termCumFreq) #initiate a blank list
    for k in range(0,len(docTermMatrix[i])): #loop through each tuple containing word + frequency pair
        tmp = docTermMatrix[i][k] #tuple
        freq[tmp[0]] = tmp[1]
    freq = np.array(freq)
    frequency_matrix["Paper" + str(i)] = freq

frequency_matrix['sum'] = frequency_matrix.loc[:,frequency_matrix.columns.str.contains('Paper')].sum(axis = 1)
frequency_matrix = frequency_matrix.sort_values(by='sum', ascending=False)

# Plot the frequency of top 20 most popular words across the corpus
top40 = frequency_matrix.reset_index().loc[0:20,['index','word','sum']].sort_values(by='sum', ascending=False)

import matplotlib.pyplot as plt

# Create a horizontal bar chart
plt.barh(top40['word'], top40['sum'], color='skyblue')
plt.xlabel('Sum')
plt.ylabel('Word')
plt.title('Top 20 Words')
plt.show()

# Question/To-do: use a set of most heavily used words per doc with threshold flexible depending on document's length

# Pass the data in LDA model
ldamodel = LDA(docTermMatrix, num_topics=4, id2word=dictionary, passes=50)

# Print out 3 words for each topic 
topics = ldamodel.show_topics(num_words=3)   
for i in range(0,len(topics)):
    print("Topic {}: {}".format(i, topics[i][1]))

# Get topic for each paper
docTopics = [ldamodel.get_document_topics(doc) for doc in docTermMatrix]

for i in range(0,len(df['filename'])):
    print("Research Paper {}: Topic {} - Probability {}".format(i, docTopics[i][0][0], docTopics[i][0][1]), end = "\n")
    
#  Populate probability of a paper belong to a topic for all 5 topics
df['Topic_0'] = float() #initialize empty variables
df['Topic_1'] = float()
df['Topic_2'] = float()
df['Topic_3'] = float()
df['Topic_4'] = float()

for i in range(0,len(df['filename'])): #loop through each paper
    topic_num_per_pdf = len(docTopics[i]) #contain n tuples associated with n topic, inside a tuple contains topic and probability score
    for k in range(0,topic_num_per_pdf): #loop through each topic k/tuple
        topic = docTopics[i][k][0] #topic number
        df.loc[i,df.columns.str.contains(str(topic))] = docTopics[i][k][1]

############################# END OF SCRIPT ##############################
##########################################################################       