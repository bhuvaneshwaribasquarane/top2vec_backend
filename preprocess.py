from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from sklearn.preprocessing import normalize


def default_tokenizer(document):
    return simple_preprocess(strip_tags(document), deacc=True)


def return_doc(doc):
    return doc


def l2_normalize(vectors):
    if vectors.ndim == 2:
        return normalize(vectors)
    else:
        return normalize(vectors.reshape(1, -1))[0]



