import json
import plotly
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    mensage_class = df.iloc[:,4:].columns
    mensage_count = df.iloc[:,4:].sum()
    
    subframe = df[['id', 'genre']].groupby('genre').agg('count').reset_index()
    mensage_genre = subframe['genre'].values
    mensage_genre_count = subframe['id'].values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=mensage_class,
                    y=mensage_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Types"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=mensage_genre,
                    y=mensage_genre_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()