import re
import nltk
from nltk.corpus import stopwords

# Download stopwords safely on Render
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)
