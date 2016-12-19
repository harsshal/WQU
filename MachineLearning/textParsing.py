__author__ = 'harsshal'

from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    vectorizer = TfidfVectorizer(
                        use_idf=True,
                        # norm=None,
                        # smooth_idf=False,
                        # sublinear_tf=False,
                        # binary=False,
                        # min_df=1,
                        # max_df=1.0, max_features=None,
                        # strip_accents='unicode',
                        # ngram_range=(1,1), preprocessor=None,
                        stop_words='english', tokenizer=None, vocabulary=None
    )
    # lots of options to play around with.few useful options I found were norm, min_df and max_df.

    for type in movie_reviews.categories():
        # only 2 categories : 'pos' and 'neg'
        type_ids = movie_reviews.fileids(type)
        X = vectorizer.fit_transform(list(movie_reviews.raw(t) for t in type_ids))
        idf = vectorizer.idf_

        # once we get weights, we just arrange it in decreasing sort
        wts = dict(zip(vectorizer.get_feature_names(), idf))
        s_wts = [(k, wts[k]) for k in sorted(wts, key=wts.get, reverse=True)]
        for key, val in s_wts[:10]:
            print(type, key, val)

if __name__ == '__main__':
    main()