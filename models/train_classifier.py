import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import sklearn.metrics as m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    """
    IN:
        database_filepath: database filepath
    OUT:
        X: messages array
        y: categories array
        columns: categories labels
    """ 
    engine = create_engine('sqlite:///%s' % (database_filepath))
    df = pd.read_sql_table('Message', engine)
    X = df.loc[:,'message']
    y = df.iloc[:,4:]
    return X, y, y.columns


def tokenize(text):
    """
    IN:
        text: messages texts
    OUT:
        messages texts tokenized and read to be classfied
    """ 
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """
    IN:
        train machine learning model
    OUT:
        grid search for best model selection
    """ 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            n_jobs=-1, 
            class_weight='balanced',
            min_samples_split=4
        )))
    ])
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__max_depth': [3, 5, None]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    IN:
        model: machine learning model trained
        X_test: sample data to be predict
        Y_test: categories of sample data
        category_names: categories labels
    OUT:
        print classification report for each category
    """
    pred = model.predict(X_test)
    pred = pd.DataFrame(pred, columns=category_names)
    for column in pred:
        print(column)
        print(m.classification_report(Y_test[column], pred[column]))
        print('*' * 30)
    


def save_model(model, model_filepath):
    """
    IN:
        model: machine learning model trained
        model_filepath: filepath to save model
    OUT:
        None
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    """
    IN:
        database_filepath: database filepath
        model_filepath: filepath to save model
    OUT:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
