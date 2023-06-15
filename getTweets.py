import twint

c = twint.Config()
c.Search = "#PrideMonth"
c.Pause = 1.0
c.Limit = 600
c.Store_csv = True
c.Output = "tweets.csv"
tweets = []

def on_tweet_callback(tweet):
    tweets.append(tweet.tweet)

c.On_tweet = on_tweet_callback

twint.run.Search(c)

# Wyświetl pobrane tweety
for tweet in tweets:
    print(tweet)