#!flask/bin/python

# Define imports
import tensorflow as tf

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from utils import JSON_MIME_TYPE
from utils import json_response
from sentence_transformers import util, models, SentenceTransformer

import json
import torch
import ray
ray.init()

# Optimize the environment for on-demand API usage
tf.compat.v1.disable_eager_execution()

graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
                                    opt_level=tf.OptimizerOptions.L1, 
                                    do_function_inlining=False)
                                )
config = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4,
            allow_soft_placement=True,
            graph_options=graph_options,
            log_device_placement=False
        )

app = Flask(__name__)
cors = CORS(app)

# Compares two sentences using Robert STS
@ray.remote
def robert_compare(s1, s2):
    # Initialze the Robert STS model
    word_embedding_model = models.Transformer("robert_sts")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Encode the two sentences and get their cosine similarity
    vect1 = model.encode(s1, convert_to_tensor=True)
    vect2 = model.encode(s2, convert_to_tensor=True)

    ret = util.pytorch_cos_sim(vect1, vect2).tolist()[0][0]

    # Clear up resources once the comparison has completed
    del word_embedding_model
    del pooling_model
    del model

    del vect1
    del vect2

    torch.cuda.empty_cache()

    # Return the results
    return ret


# Processes a request to compare two sentences using Robert STS
@app.route('/robert', methods=['POST'])
def robert_comparer():
    result = "0"

    # Ensure that the request data is in JSON format
    if request.content_type != JSON_MIME_TYPE:
        error = json.dumps({'error': 'Invalid Content Type'})
        response = json.dumps({'result': str(result)})
        return json_response(response, 200)

    # Ensure that the first sentence (s1) and second sentence (s2) are in the request data
    data = request.get_json()
    if not all([data.get('s1'), data.get('s2')]):
        error = json.dumps({'error': 'Missing fields'})
        response = json.dumps({'result': str(result)})
        return json_response(response, 200)

    # Extract the sentences from the request data
    s1 = data['s1']
    s2 = data['s2']

    print("{" + s1 + "},{" + s2 + "}")

    # Compare the two sentences using Robert STS
    result = robert_compare.remote(s1, s2)
    result = ray.get(result)

    # Ensure that the result is within the range of 0 to 1
    if result >= 0 and result <= 1:
        result = str(result)
    elif result > 1:
        result = 1.0
    else:
        result = "0"

    # Return the comparison result
    response = json.dumps({'result': str(result)})

    print(response)
    return json_response(response, 200)



if __name__ == '__main__':
    # Launch the API
    app.run(host='127.0.0.1', port='12345', debug=False)