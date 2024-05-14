import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

from textblob import TextBlob

df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)
#print(df.head())
# inputting the data file containing the reviews. Printing to check correct input

reviews_data = df['reviews.text'] # Only to look at the reviews.text column.

clean_data = df.dropna(subset=['reviews.text']) # Removing any blank rows in the reviews.text column.
#print(df.head(10)) # checking that no blank rows are there.

clean_data.count()
# Checking the number of rows in the data set to know the limits when choosing my data sample.

def analyse_polarity(text):
    # Preprocess the text with spaCy
    doc = nlp(text.lower())
    filtered_doc = [token.text for token in doc if not token.is_stop]

    # Analyze sentiment with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    return polarity
    
# Comparing two different reviews.

my_review_choice1 = clean_data['reviews.text'][50]
print(my_review_choice1)
my_review_choice2 = clean_data['reviews.text'][1500]
print(my_review_choice2)

query_doc = nlp(my_review_choice2)

# Compares the similarity between to selected reviews from the data set to 4DP.

for text in my_review_choice1:
    review = nlp(my_review_choice1)
    similarity = query_doc.similarity(review)
    scores = round(similarity, 4)

print(f"Similarity score: {str(scores)}")

# Selecting a sample from the data set
my_review_choice_range = clean_data['reviews.text'][0:1000]

# Starting a counter for the number of positive, negative and neutral reviews.

neg_count = 0
pos_count = 0
neut_count = 0

for text in my_review_choice_range:

    polarity_score = analyse_polarity(text)

# Using the function to create a polarity score for each individual review.
# the count tracks how many positive, negative and neutral reviews are in the selected range for the data set.

    if polarity_score >0:
        sentiment = "Positive"
        pos_count = pos_count + 1
        #print(f"Text: {text}\nPolarity Score: {polarity_score}\nSentiment: {sentiment}")
    elif polarity_score < 0:
        sentiment = "Negative"
        neg_count = neg_count + 1
        # print(f"Text: {text}\nPolarity Score: {polarity_score}\nSentiment: {sentiment}")
    else:
        sentiment = "Neutral"
        neut_count = neut_count + 1
        #print(f"Text: {text}\nPolarity Score: {polarity_score}\nSentiment: {sentiment}")

# Gives an overall view of positive, negative and neutral reviews within the selected range of the data.
print(f"Positive Reviews:  {str(pos_count)}")
print(f"Negative Reviews: {str(neg_count)}")
print(f"Neutral Reviews: {str(neut_count)}")