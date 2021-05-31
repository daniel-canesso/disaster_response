# Disaster Response Pipeline Project

### Motivation:

The Disaster Response project aims to create a machine learning model capable of correctly categorizing messages sent by social networks to the competent bodies during periods of humanitarian, environmental and social crises.

With this model, the authorities will be able to orient themselves in a more adequate way, providing correct and quick assistance, according to the countless requests that arrive at these moments.

In addition to all this, classified messages may be sent to the correct bodies, so that assistance also comes from the right entity.

### File Structure:

- app
- | - template
- | |- master.html # main page of web app
- | |- go.html # classification result page of web app
- |- run.py # Flask file that runs app
- data
- |- disaster_categories.csv # data to process
- |- disaster_messages.csv # data to process
- |- process_data.py # Python file that process data and save (ETL)
- |- InsertDatabaseName.db # database to save clean data to
- models
- |- train_classifier.py # Python file that train and save classifier
- |- classifier.pkl # saved model
- README.md


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
