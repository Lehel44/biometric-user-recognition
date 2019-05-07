# -*- coding: utf-8 -*-

from flask import Flask, request, Response, jsonify
import tensorflow as tf
from keras.models import load_model
import os
import io
import numpy as np
import logging
import soundfile as sf
from pprint import pformat
from werkzeug.utils import secure_filename
from preprocess import BatchPreProcessor, preprocess_instances
import uuid
import pickle
from config import Config

# Flask app.
app = Flask(__name__)

# Load configuration.
app.config.from_object('config.Config')

# Mute excessively verbose Tensorflow output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Load keras modell.
model = load_model(app.config['SIAMESE_MODEL_PATH'])
model._make_predict_function()
graph = tf.get_default_graph()


def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_audio_file(file_id):
  if file_id not in request.files:
    return False
  file = request.files[file_id]
  if file and allowed_file(file.filename):
    return True
  return False

def save_audio_file(file_id, path):
  username = request.form.get('username')
  userid = str(uuid.uuid4())
  file = request.files[file_id]
  file.save(os.path.join(path, userid + '.wav'))
  return userid, username

@app.route('/enroll', methods=['POST'])
def enroll():
  if check_audio_file('audio_file'):
    userid, username = save_audio_file('audio_file', app.config['UPLOAD_FOLDER'])
    # Save user data.
    user_dict = dict()
    user_dict_path = app.config['USER_DICT_PATH']
    try:
      user_dict = pickle.load(open(user_dict_path, 'rb'))
      user_dict[userid] = username
      pickle.dump(user_dict, open(user_dict_path, 'wb'))
    except (OSError, IOError) as e:
      user_dict[userid] = username
      pickle.dump(user_dict, open(user_dict_path, 'wb'))

    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], userid + '.wav')
    preprocess(userid, audio_path, 3, 2)
  else:
    return Response(str(400), mimetype='text/plain')

  return Response(str(200), mimetype='text/plain')

@app.route('/identify', methods=['POST'])
def identify():
  if check_audio_file('audio_file'):
    userid, _ = save_audio_file('audio_file', app.config['IDENTIFY_PATH'])
    audio_path = os.path.join(app.config['IDENTIFY_PATH'], userid + '.wav')
    preprocess(userid, audio_path, 3, 2)
    response = predict(userid)
    app.logger.info(pformat(response))
    user_dict = pickle.load(open(app.config['USER_DICT_PATH'], 'rb'))
    app.logger.info(pformat(user_dict))
    return Response(str(200), mimetype='text/plain')
  return Response(str(400), mimetype='text/plain')

'''Preprocesses raw audio files in the enrolling phase. Preprocessing
   includes standardization and downsampling of the current audio sample.'''
def preprocess(userid, audio_file_path, sample_length, downsampling):
  instance, sample_rate = sf.read(audio_file_path)
  # Cut 3 second
  middle = int(len(instance)/2)
  dist = sample_length * sample_rate
  instance = instance[middle-dist/2:middle+dist/2]
  # Expand to 3 dimension.
  input = np.stack([instance])[:, :, np.newaxis]
  batch_preprocessor = BatchPreProcessor('classifier', preprocess_instances(downsampling))
  (input, _) = batch_preprocessor((input, []))
  # Save preprocessed file.
  np.save(os.path.join(app.config['PREPROCESSED_PATH'], userid), input)

def predict(current_userid):
  # Get preprocessed audio samples.
  all_audio_data = []
  all_audio_names = []
  current_audio_data = []
  user_dict = pickle.load(open(app.config['USER_DICT_PATH']))
  for userid in user_dict.keys():
    path = os.path.join(app.config['PREPROCESSED_PATH'], userid + '.npy')
    file = np.load(path)
    all_audio_data.append(file)
    all_audio_names.append(user_dict[userid])

  current_audio_data = np.load(os.path.join(app.config['PREPROCESSED_PATH'], current_userid + '.npy'))

  num_of_speakers = len(all_audio_data)

  input_1 = np.stack([current_audio_data[0]] * num_of_speakers)
  input_2 = np.concatenate(all_audio_data)

  with graph.as_default():
    result = model.predict([input_1, input_2])

  print(pformat(result))
  print(pformat(all_audio_names))
  response = np.concatenate(result).tolist()
  return response


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
