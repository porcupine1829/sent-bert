import re
import html
import pandas as pd

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_text(text):
    """Efficiently clean text by removing HTML tags, quotes, and normalizing whitespace."""
    if pd.isna(text) or not text:
        return text
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(' ', text)
    text = text.replace('"', '').replace("'", '')
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    return text

def clean_dataframe_column(df, column_name):
    """Efficiently clean a DataFrame column."""
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    return df

df = pd.read_csv("data\\imdb-reviews-data.csv")

mapping = {"negative": 0, "positive": 1}
df["label"] = df["label"].map(mapping)

df = clean_dataframe_column(df, 'text')

train = df.sample(frac=0.8, random_state=42)  # 80% train
val = df.drop(train.index)                    # 20% validation

train.to_csv("data\\processed\\train.csv", index=False)
val.to_csv("data\\processed\\validation.csv", index=False)


