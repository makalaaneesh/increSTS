from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
def get_sentiment_score(text):
	analyzer = SentimentIntensityAnalyzer()
	jsonscore = analyzer.polarity_scores(text)
	return jsonscore["compound"]



# print get_sentiment_score("omg her dress is so amazing")