from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from preprocess import *
from datetime import datetime


class MLModel:
    def __init__(self, documents):
        self.documents = documents
        self.np_docs = np.array(documents, dtype="object")
        self.document_ids = np.array(range(0, len(documents)))
        self.doc_id_map = dict(zip(self.document_ids, documents))
        self.pretrained_model = None
        self.document_embeddings = None
        self.umap_embeddings_cluster = None
        self.umap_data_viz = None
        self.clusters = None
        self.topic_vectors = None

    def set_pretrained_model(self, model_path, document_embeddings):
        self.pretrained_model = SentenceTransformer(model_path)
        self.document_embeddings = np.load(document_embeddings)
        if model_path == "all-MiniLM-L6-v2":
            self.model_name = "default"
        else:
            self.model_name = model_path

    def preprocess_documents(self):
        tokenizer = default_tokenizer
        tokenized_corpus = [tokenizer(doc) for doc in self.documents]
        vectorizer = CountVectorizer(tokenizer=return_doc, preprocessor=return_doc)
        doc_word_counts = vectorizer.fit_transform(tokenized_corpus)
        words = vectorizer.get_feature_names()
        word_counts = np.array(np.sum(doc_word_counts, axis=0).tolist()[0])
        vocab_inds = np.where(word_counts > 50)[0]
        if len(vocab_inds) == 0:
            raise ValueError(f"A min_count of  results in "
                             f"all words being ignored, choose a lower value.")
        self.vocab = [words[ind] for ind in vocab_inds]
        self.word_indexes = dict(zip(self.vocab, range(len(self.vocab))))
        self.word_vectors = l2_normalize(np.array(self.pretrained_model.encode(self.vocab, show_progress_bar=True)))

    def reduce_dims(self):
        self.umap_embeddings_cluster = umap.UMAP(n_neighbors=25,
                                                 n_components=5,
                                                 metric='cosine',
                                                 random_state=42).fit_transform(self.document_embeddings)

        # Visualization by UMAP reduction further to 2d
        self.umap_data_viz = umap.UMAP(n_neighbors=25, n_components=2, min_dist=0.5, metric='cosine',
                                       random_state=42).fit_transform(
            self.document_embeddings)

    def make_clusters(self, min_cluster_size):
        # HDBSCAN Clustering
        self.clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        metric='euclidean',
                                        cluster_selection_method='eom').fit(self.umap_embeddings_cluster)

        cluster_labels = self.clusters.labels_
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = l2_normalize(
            np.vstack([self.document_embeddings[np.where(cluster_labels == label)[0]]
                      .mean(axis=0) for label in unique_labels]))

        self.topic_words, self.topic_word_scores = self.find_topic_words_and_scores(topic_vectors=self.topic_vectors)
        self.doc_top, self.doc_dist = self.calculate_documents_topic(self.topic_vectors, self.document_embeddings)
        self.doc_top = np.array(self.doc_top, dtype="int")

    def get_topic_words(self, topic_number):
        return self.topic_words[topic_number][:10], self.topic_word_scores[topic_number][:10]

    def get_document_topic(self, doc_id):
        return self.doc_top[doc_id]

    def render_plot_data(self):
        #hue_cat = [self.topic_words[top][0] for top in self.doc_top]
        hue_cat = [int(top) for top in self.doc_top]
        #sns.scatterplot(x=self.umap_data_viz[:, 0], y=self.umap_data_viz[:, 1], hue=hue_cat, palette="Paired")
        # sns.scatterplot(x=[umap_data[recluster_point][0]], y=[umap_data[recluster_point][1]], color='red', s=100)

        return self.umap_data_viz[:, 0], self.umap_data_viz[:, 1], hue_cat, [str(a)+" - "+str(b[:100]) for a,b in zip(self.document_ids, self.documents)]

    def render_selected_topic(self, topics):
        hue_cat = [self.topic_words[top][0] for top in self.doc_top]
        cluster_labels = pd.Series(self.clusters.labels_)
        s = (cluster_labels == topics[0])
        return self.umap_data_viz[s][:, 0], self.umap_data_viz[s][:, 1], self.topic_words[topics[0]][0], [str(a)+" - "+str(b[:100]) for a,b in zip(self.document_ids[s], self.np_docs[s])]

    def search_documents_by_topic(self, topic_num, num_docs=10, return_documents=True):

        # self._validate_topic_num(topic_num, reduced)
        # self._validate_topic_search(topic_num, num_docs, reduced)

        topic_document_indexes = np.where(self.doc_top == topic_num)[0]
        topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist[topic_document_indexes]))
        doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
        doc_scores = self.doc_dist[doc_indexes]
        doc_ids = self.document_ids[doc_indexes]

        if self.documents is not None and return_documents:
            documents = self.np_docs[doc_indexes]
            doc_topics = self.doc_top[doc_indexes]
            return documents, doc_topics, doc_ids
        else:
            return doc_scores, doc_ids

    def find_topic_words_and_scores(self, topic_vectors):
        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.word_vectors)
        top_words = np.flip(np.argsort(res, axis=1), axis=1)
        top_scores = np.flip(np.sort(res, axis=1), axis=1)

        for words, scores in zip(top_words, top_scores):
            topic_words.append([self.vocab[i] for i in words[0:50]])
            topic_word_scores.append(scores[0:50])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

    def calculate_documents_topic(self, topic_vectors, document_vectors, dist=True, num_topics=None):
        batch_size = 10000
        doc_top = []
        if dist:
            doc_dist = []

        if document_vectors.shape[0] > batch_size:
            current = 0
            batches = int(document_vectors.shape[0] / batch_size)
            extra = document_vectors.shape[0] % batch_size

            for ind in range(0, batches):
                res = np.inner(document_vectors[current:current + batch_size], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

                current += batch_size

            if extra > 0:
                res = np.inner(document_vectors[current:current + extra], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])
            if dist:
                doc_dist = np.array(doc_dist)
        else:
            res = np.inner(document_vectors, topic_vectors)

            if num_topics is None:
                doc_top = np.argmax(res, axis=1)
                if dist:
                    doc_dist = np.max(res, axis=1)
            else:
                doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                if dist:
                    doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

        if num_topics is not None:
            doc_top = np.array(doc_top)
            if dist:
                doc_dist = np.array(doc_dist)

        if dist:
            return doc_top, doc_dist
        else:
            return doc_top

    def _get_combined_vec(self, vecs, vecs_neg):
        combined_vector = np.zeros(self.document_embeddings.shape[1], dtype=np.float64)
        for vec in vecs:
            combined_vector += vec
        for vec in vecs_neg:
            combined_vector -= vec
        combined_vector /= (len(vecs) + len(vecs_neg))
        combined_vector = self._l2_normalize(combined_vector)
        return combined_vector

    def search_documents_by_keywords(self, keywords, num_docs, keywords_neg=None, return_documents=True,
                                     use_index=False, ef=None):
        if keywords_neg is None:
            keywords_neg = []
        # self._validate_num_docs(num_docs)
        # keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)
        word_vecs = self.word_vectors[[self.word_indexes[word] for word in keywords]]
        neg_word_vecs = self.word_vectors[[self.word_indexes[word] for word in keywords_neg]]

        combined_vector = self._get_combined_vec(word_vecs, neg_word_vecs)
        doc_indexes, doc_scores = self._search_vectors_by_vector(self.document_embeddings,
                                                                 combined_vector, num_docs)
        doc_ids = self.document_ids[doc_indexes]

        if self.documents is not None and return_documents:
            documents = self.np_docs[doc_indexes]
            doc_topics = self.doc_top[doc_indexes]
            return documents, doc_topics, doc_ids
        else:
            return doc_scores, doc_ids


    @staticmethod
    def _l2_normalize(vectors):

        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]

    @staticmethod
    def _search_vectors_by_vector(vectors, vector, num_res):
        ranks = np.inner(vectors, vector)
        indexes = np.flip(np.argsort(ranks)[-num_res:])
        scores = np.array([ranks[res] for res in indexes])

        return indexes, scores

    def prepare_retrain_dataset(self, doc_id, source_cluster, target_cluster):
        cluster_labels = pd.Series(self.clusters.labels_)
        ss = (cluster_labels == source_cluster)
        source_cluster_docs = self.np_docs[ss]

        ts = (cluster_labels == target_cluster)
        target_cluster_docs = self.np_docs[ts]
        feedback_point = self.documents[doc_id]

        train_examples = []

        for doc in target_cluster_docs[:20]:
            train_examples.append(InputExample(texts=[feedback_point, doc], label=0.9))

        for doc in source_cluster_docs[:20]:
            train_examples.append(InputExample(texts=[feedback_point, doc], label=0.1))

        return train_examples

    def retrain_embeddings(self, retrain_points):
        full_train_data = []
        for i in retrain_points:
            full_train_data.extend(self.prepare_retrain_dataset(int(i["document_id"]), int(i["sourceCluster"]), int(i["targetCluster"])))
        retrain_model = SentenceTransformer('all-MiniLM-L6-v2')
        train_dataloader = DataLoader(full_train_data, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(retrain_model)

        # Tune the model
        retrain_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
        embeddings_retrained = retrain_model.encode(self.documents, show_progress_bar=True)

        curr_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        np.save('saved/doc-embeddings_'+curr_time, embeddings_retrained)
        retrain_model.save("saved/retrained-model_"+curr_time)

    def run_all(self, model_path, documents_embeddings, min_cluster_size):
        self.min_cluster_size = min_cluster_size
        self.set_pretrained_model(model_path, documents_embeddings)
        self.preprocess_documents()
        self.reduce_dims()
        self.make_clusters(min_cluster_size)


# newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# model_instance = MLModel(newsgroups_train.data, 'all-MiniLM-L6-v2', 'doc-embeddings_default.npy')
# model_instance.run_all()
