from emoji import demojize
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

tokenizer = TweetTokenizer()
stemmer = SnowBallStemmer()
def preprocess(tweet, stemmer=None):
    stop_words = set(stopwords.words('english'))
    tweet = str(tweet).lower()
    tweet = re.sub(r"what's", "what is ", tweet)
    tweet = re.sub(r"\'s", " ", tweet)
    tweet = re.sub(r"\'ve", " have ", tweet)
    tweet = re.sub(r"n't", " not ", tweet)
    tweet = re.sub(r"i'm", "i am ", tweet)
    tweet = re.sub(r"\'re", " are ", tweet)
    tweet = re.sub(r"\'d", " would ", tweet)
    tweet = re.sub(r"\'ll", " will ", tweet)
    tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(https\S+|www\S+|http\S+)|([^A-Za-z\s])|([0-9])", " ", tweet)
    processed_tweet = " ".join(y for y in word_tokenize(tweet) if y not in stop_words)
    if stemmer != None: processed_tweet = " ".join(
        stemmer.stem(y) for y in word_tokenize(tweet) if y not in stop_words)
    return processed_tweet

"""
Normalize Token from https://github.com/VinAIResearch/BERTweet.git
"""
def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

"""
NormalizeTweet from https://github.com/VinAIResearch/BERTweet.git
"""
def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

