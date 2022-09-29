
import numpy as np
import pandas as pd

data = pd.read_excel("./data/qanda.xlsx")
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import re

def preprocess_sentences(input_sentences):
    return [input_sentence
            for input_sentence in input_sentences]
        
module = hub.load('./universal-sentence-encoder-multilingual-qa_3')

# # Create response embeddings

# Create encodings for test questions
"""
I am using using Universal Encoder Multilingual Q&A model for question-answer retrieval of text, illustrating the use of question_encoder and response_encoder of the model. We use sentences from SQuAD paragraphs as the demo dataset, each sentence and its context 
(the text surrounding the sentence) is encoded into high dimension embeddings with the response_encoder. These embeddings are stored in an index built using the simpleneighbors library for question-answer retrieval.
On retrieval a random question is selected from the SQuAD dataset and encoded into high dimension embedding with the question_encoder and query the simpleneighbors index returning a list of approximate nearest neighbors in semantic space.
"""

"""
Model Details
1. Transformer
2. Covers 16 languages, strong performance on cross-lingual question answer retrieval.
3. Use the question_encoder signature to encode variable length questions in any of the aforementioned languages and the output is a 512 dimensional vector. The default signature is identical with the question_encoder signature.
4. Use the response_encoder signature to encode the answer and the output is a 512 dimensional vector.
5. The response_encoder signature acceptes two input fields:
    5 a. text: the answer text.
    5 b.context: usually the text around the answer text, for example it could be 2 sentences before plus 2 sentences after, it could also be the paragraph containing the answer text. If you don't have context to include, you can duplicate of answer into this field.
6. All input text can have arbitrary length! However, model time and space complexity is $$O(n^2)$$ for question and response input length $$n$$ and $$O(n)$$ for context length. We recommend question and response inputs that are approximately one sentence in length.

download model: https://drive.google.com/file/d/1QubywTsnto6ZEIM6geeDgqqFrOgDt5S6/view?usp=sharing
"""
def gentral_question_response(ip):
    response_encodings = module.signatures['response_encoder'](
            input=tf.constant(preprocess_sentences(data.Context)),
            context=tf.constant(preprocess_sentences(data.Answer)))['outputs']
    question_encodings = module.signatures['question_encoder'](
        tf.constant(ip)
    )['outputs']

    # Get the responses
    test_responses = data.Answer[np.argmax(np.inner(question_encodings, response_encodings))]
    return test_responses
print(gentral_question_response("fertlizer"))


