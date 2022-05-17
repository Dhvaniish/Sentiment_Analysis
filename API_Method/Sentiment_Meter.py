import os
import yagmail
from google.cloud import language_v1
import mysql.connector
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="website"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM reviews ORDER BY id DESC LIMIT 1")

myresult = mycursor.fetchone()

text = myresult[5]

  
# GCloud Auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/dhvan/Downloads/sentiment-analysis-350404-2e3172886e42.json"

# Email Credentials
yag = yagmail.SMTP('dripppypapi@gmail.com', '#MUKkP7@')

client = language_v1.LanguageServiceClient()

# The text to analyze
text
document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment = client.analyze_sentiment(request={"document": document}).document_sentiment


def analyze():

    print("Text: {}".format(text))
    print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
    # [END language_quickstart]

    if sentiment.score > 0.3:
        score = "Positive"
    elif sentiment.score < -0.3:
        score = "Negative"
    elif -0.3 < sentiment.score < 0.3:
        score = "Neutral"
        
    print(score + " Review")


if __name__ == "__main__":
    analyze()
    
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(500,250)
    w.setWindowTitle("Sentiment Analysis")
    
    label1 = QLabel(w)
    label1.setText("Review: " + text)
    label1.setAlignment(Qt.AlignCenter)
    label1.move(50,50)
    label1.show()
    
    label2 = QLabel(w)
    sentiment.score
    string_value = str(sentiment.score)
    label2.setText("Sentiment Meter: " + string_value)
    label2.setAlignment(Qt.AlignCenter)
    label2.move(50,100)
    label2.show()
    
    if sentiment.score > 0.3:
        score = "Positive"
    elif sentiment.score < -0.3:
        score = "Negative"
    elif -0.3 < sentiment.score < 0.3:
        score = "Neutral"
        
    label3 = QLabel(w)
    label3.setText("Sentiment Prediction: " + score)
    label3.setAlignment(Qt.AlignCenter)
    label3.move(50,150)
    label3.show()
 
    w.show()
    sys.exit(app.exec_())
    