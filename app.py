from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

tfidf_vectorizer = pickle.load(open('sentimental_analysis_models/tfidf_vectorizer.pkl', 'rb'))
logreg = pickle.load(open('sentimental_analysis_models/log.pkl', 'rb'))

# downloading the NLTK data
nltk_data_dir = './nltk_data'
nltk.data.path.append(nltk_data_dir)

wordnet = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in stop_words]
    return ' '.join(review)

def find_review_column(columns): 
    for col in columns:
        if 'review' in col.lower() or 'text' in col.lower():
            return col
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    
    X = tfidf_vectorizer.transform([review])
    sentiment = logreg.predict(X)[0]
    
    sentiment_label = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    
    return jsonify({'sentiment': sentiment_label[sentiment]})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        
        try:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return jsonify({'error': 'Unable to read the CSV file with standard encodings.'}), 400
            
            review_col = find_review_column(df.columns) # Find the column with review text
            if not review_col:
                return jsonify({'error': 'CSV must contain a column with review text.'}), 400
            
            df['Cleaned_Review_Text'] = df[review_col].apply(preprocess_text)
            X = tfidf_vectorizer.transform(df['Cleaned_Review_Text'])
            df['Sentiment'] = logreg.predict(X)
            
            sentiment_label = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
            df['Sentiment_Label'] = df['Sentiment'].map(sentiment_label)
            
            output_filename = 'sentiment_results.csv'
            output_path = os.path.join('/tmp', output_filename)
            df.to_csv(output_path, index=False)
            
            return send_file(output_path, as_attachment=True, download_name=output_filename)
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)