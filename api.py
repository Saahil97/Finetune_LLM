from flask import Flask
from flask_restful import Api
from flask_cors import CORS
#from authenticate_voice import AuthenticateVoice
from inference import Trivia


app = Flask(__name__)
CORS(app)
#cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)
app.json.sort_keys=False


@app.route('/check')
def index():
    return "<h1>Flask API Server is working for WWE_Trivia</h1>"

"""
    Routes
    ** '/ai/orderedresponse'

"""


api.add_resource(Trivia, '/generation2')


# api.add_resource(Todo, '/todos/<todo_id>')

# api.__init__(app)

if __name__ == '__main__':
    app.run(port=5081, host='0.0.0.0')