import os
import json
import logging
import logging.handlers
from datetime import datetime
import re

from flask import Flask, request, jsonify
from flask_restful import reqparse, Resource
from langchain.llms import LlamaCpp
#from sentence_transformers import SentenceTransformer, util
from langchain.callbacks import StdOutCallbackHandler
import llama_cpp 
from llama_cpp import Llama

import numpy as np
import re
import random

def load_llm():
    """
    Load the Large Language Model for question generation.
    nknk
    Returns:
        llm: Initialized language model object.
    """
    model_path=" Take model from here: https://huggingface.co/Saahil97/mistral_wwetrivia"
    #if os.environ.get("MODEL_DOWNLOAD_PATH"):
    #    model_path =os.path.join(os.environ.get("MODEL_DOWNLOAD_PATH"), model_path)
    try:
        llm = Llama(
            model_path=model_path,
            verbose=True,
            n_ctx=4000,
            n_batch=1024,
            n_threads=16,
            n_gpu_layers=-1,
            
        )
        return llm
    except Exception as e:
        return None
    
llm = load_llm()


def process_quiz_text(text):
    question_start = text.find("Question:") + len("Question: ")
    question_end = text.find("\nOptions:")
    question = text[question_start:question_end].strip()

    options_start = text.find("Options:") + len("Options:\n")
    options_end = text.find("\nCorrect:")
    options_text = text[options_start:options_end].strip()

    # Split the options into a list
    options = options_text.split("\n")

    correct_start = text.find("Correct:") + len("Correct: ")
    correct_end = text.find("\n", correct_start)
    correct_answer = text[correct_start:correct_end].strip()

    options_with_correctness = [(option, option.endswith(correct_answer)) for option in options]

    # Shuffle the options
    random.shuffle(options_with_correctness)

    correct_label = None

    # Rebuild the shuffled options list and find the new correct option
    shuffled_options = []
    for i, (option, is_correct) in enumerate(options_with_correctness):
        # Assign new option labels (a, b, c, d)
        new_option = f"{chr(97 + i)}: {option[3:].strip()}"
        shuffled_options.append(new_option)
        if is_correct:
            correct_label = chr(97 + i)  # Track the new correct label (a, b, c, d)

        new_response = {
            "question": question,
            "options": shuffled_options,
            "correct_answer": f"{correct_label}: {correct_answer}"
        }

    return new_response

class Trivia(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('context', type=str, required=True, help='Context is required')
        args = parser.parse_args()

        context = args.context
        query = f'''
            {context}
            
            You are a subject matter expert and you are tasked to generate Multiple Choice Questions related to WWE. 
            Generate a question with 1 correct answer and 3 wrong answers. DO NOT REPEAT SAME NAMES.           
            
            '''
        try:
            if not llm:
                return {"Result": False, "Message": "LLM not loaded"}, 500

            result = llm(query, max_tokens=8000)
            result = result['choices'][0]
            res = result['text']
            
            formatted = process_quiz_text(res)
            return formatted
        
        except Exception as e:
            return {"Result": False, "Message": str(e)}, 500
