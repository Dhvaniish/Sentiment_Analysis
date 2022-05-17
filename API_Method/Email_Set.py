import os
import yagmail
from google.cloud import language_v1
import mysql.connector
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
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

def email():
    
    if sentiment.score > 0.3:
        score = "Positive"
    elif sentiment.score < -0.3:
        score = "Negative"
    elif -0.3 < sentiment.score < 0.3:
        score = "Neutral"
        
    score
    to = 'dhvanish.spam@gmail.com'
    subject = 'New Review Posted'
    # body = 'This is obviously the body'
    html = '''
    <html>
    <font color="black">

        <body>
            <h2>Feedback Report</h1>
                <p>A new review was posted for the product: Apple Watch</p>
                <p>The review seems to be relatively ''' + score + '''.</p>
        </body>
    </font>
    </html>
    '''
    yag.send(to=to, subject=subject, contents=[html])


if __name__ == "__main__":

    email()
    
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(500,100)
    w.setWindowTitle("Email Sent")
    
    label1 = QLabel(w)
    label1.setText("Email sent to Seller")
    label1.setAlignment(Qt.AlignCenter)
    label1.move(100,35)
    label1.show()
    
    w.show()   
    sys.exit(app.exec_())