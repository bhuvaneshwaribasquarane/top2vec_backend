import collections
import os
import time

import rq
from flask import Flask, url_for, Response
from flask import request, redirect
from flask import render_template, jsonify
from flask_cors import CORS
from rq.job import Job
from worker import conn
import json
FLASK_APP = Flask(__name__)
CORS(FLASK_APP)
tasksQueue = rq.Queue(connection=conn, default_timeout=3600)

FLASK_APP.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./ir_classifier.db'
FLASK_APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
from model import *
global_model = None
DATASET = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data
global_jobs_list = {}


@FLASK_APP.route("/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task = tasksQueue.fetch_job(task_id)

    if task:
        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id(),
                "task_status": task.get_status(),
                "task_result": task.result,
            },
        }
    else:
        response_object = {"status": "error"}

    response = json.dumps(response_object)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/progress/<string:job_id>')
def progress(job_id):
    def get_status():

        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()

        while status != 'finished':

            status = job.get_status()
            job.refresh()

            d = {'status': status}

            if 'progress' in job.meta:
                d['value'] = job.meta['progress']
            else:
                d['value'] = 0

            # IF there's a result, add this to the stream
            if job.result:
                d['result'] = job.result

            json_data = json.dumps(d)
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    return Response(get_status(), mimetype='text/event-stream')


@FLASK_APP.route('/check_model_status')
def check_model_status():
    if global_model is None:
        response = {"status": "Not ready", "model_name": ""}
    else:
        response = {"status": "Ready", "model_name": global_model.model_name,
                    "last_run_min_cluster": global_model.min_cluster_size}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/get_model')
def get_model():
    response = {'saved': next(os.walk('saved/'))[1]}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/set_model/<string:model_name>/<int:min_cluster_size>')
def set_model(model_name, min_cluster_size):
    global global_model
    global_model = MLModel(DATASET)
    if model_name == 'default':
        global_model.run_all('all-MiniLM-L6-v2', 'saved/doc-embeddings_default.npy', min_cluster_size)
    else:
        time_stamp = model_name.split('_')[1]
        global_model.run_all('saved/retrained-model_'+time_stamp, 'saved/doc-embeddings_'+time_stamp+'.npy', min_cluster_size)
        
    return {"message": "model loaded successfully"}


@FLASK_APP.route('/topics')
def topics():
    response = {idx: topic[0] for idx, topic in enumerate(global_model.topic_words)}
    response = [{"topic_number": ids, "topic_word": val}for ids, val in response.items()]
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/full_plot')
def full_plot():
    x, y, hue, text = global_model.render_plot_data()
    response = {'x': x.tolist(), 'y': y.tolist(), 'hue': hue, 'text': text}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/single_topic_plot/<string:topics>')
def single_topic_plot(topics):
    topics = [int(top) for top in topics.split(',')]
    x, y, hue, text = global_model.render_selected_topic(topics)
    response = {'x': x.tolist(), 'y': y.tolist(), 'hue': hue, 'text': text}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/topic_documents/<string:topic_num>')
def topic_documents(topic_num):

    documents, doc_topics, doc_ids = global_model.search_documents_by_topic(int(topic_num), 1000000)
    response = {'documents': [doc for doc in documents.tolist()], 'doc_topics': doc_topics.tolist(),
                'doc_ids': doc_ids.tolist()}

    data = []
    for i in range(len(documents.tolist())):
        data.append({'doc_id': response['doc_ids'][i],
                     'topic_id': response['doc_topics'][i],
                     'document': response['documents'][i]})
    response = json.dumps(data)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/topic_words/<int:topic_num>')
def get_topic_words(topic_num):
    topic_words, topic_scores = global_model.get_topic_words(topic_num)
    response = {'topic_words': topic_words.tolist()[::-1], 'topic_scores': topic_scores.tolist()[::-1]}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/document_topic/<int:doc_id>')
def get_document_topic(doc_id):
    topic_id = global_model.get_document_topic(doc_id)
    response = {'topic_id': int(topic_id)}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/document_distribution')
def document_distribution():
    doc_top = global_model.doc_top
    response = {'distribution': collections.Counter(doc_top)}
    response = {str(key): str(value) for key, value in response['distribution'].items()}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/document_topic_scores')
def document_topic_scores():
    doc_top = global_model.doc_top
    response = {'distribution': collections.Counter(doc_top)}
    response = {str(key): str(value) for key, value in response['distribution'].items()}
    response = json.dumps(response)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/search_keywords/<string:keywords>')
def search_keywords(keywords):
    keywords = keywords.split(',')
    documents, doc_topics, doc_ids = global_model.search_documents_by_keywords(keywords, 100)
    documents, doc_topics, doc_ids = documents, doc_topics, doc_ids
    response = {'documents': [doc[:250] for doc in documents.tolist()], 'doc_topics': doc_topics.tolist(),
                'doc_ids': doc_ids.tolist()}
    data = []
    for i in range(len(documents.tolist())):
        data.append({'doc_id': response['doc_ids'][i],
                     'topic_id': response['doc_topics'][i],
                     'document': response['documents'][i]})
    response = json.dumps(data)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/retrain', methods=['POST'])
def retrain_embeddings():
    retrain_points = json.loads(request.data)
    job = tasksQueue.enqueue(global_model.retrain_embeddings, retrain_points, result_ttl=-1)
    global global_jobs_list
    global_jobs_list[job.get_id()] = "started"
    response = json.dumps(global_jobs_list)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/download_predict', methods=['POST'])
def download_predict():
    username = request.form['username']
    predict = pd.read_csv("predict_"+username)

    return Response(predict.to_csv(), status=200, mimetype='text/csv')


if __name__ == '__main__':
    FLASK_APP.run(port=9000)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
