from fastapi.middleware.cors import CORSMiddleware
from nltk.util import pr
from supertokens_fastapi import get_cors_allowed_headers
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from general_qa import *
import json
import random
from wornet import *
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation,remove_stopword,check_spelling
from text_preprocessing import preprocess_text
preprocess_functions = [to_lower,remove_punctuation,remove_stopword]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"] + get_cors_allowed_headers(),
)

s=set(range(6))
q = ['when to obtain gst registration',
 'documents  for  gst registration',
 'what is gstr 1',
 'what is gstr3b',
 'what is gstr9',
 'what is gstr4 ',
 '\nwhat is gstr2a',
 'what is gstr2b ',
 'due dates for gst returns',
 'penalties under gst',
 'efresh experts for gst',
 'show efresh gst expert contact details',
 'how many members are required to form a producer company',
 'what is the minimum number of directors to form a producer company',
 'who can be member in a producer company ',
 'minimum capital required to incorporate a producer company',
 'incorporation of a producer company ',
 'what are the benefits derived by the members of producer companies',
 'whether appointment of chief executive is mandatory',
 'what are the powers and functions of chief executive',
 'board meetings',
 'annual general body meeting',
 'annual general body meeting notice',
 'quorum for general body meetings including agm',
 'books of account',
 'internal audit',
 'transferability of shares and attendant rights',
 'voting rights of members',
 'efresh experts for companies act ',
 'show efresh companies act expert contact details']
import json

with open('./data/question_sugestion_faq.json', 'r') as nm:
    data2 = json.load(nm)

"""
This /bot is main function api route here i am getting two values from front end "txt" and "id".i use key "txt" to geeting user query
value and "id" is uuid this id is gentrate by front end after i get this 2 value i forward it to main() function.
please check application_bot in import section 
"""
"""
if id 1 i am going to show GST related questions if id 2 company_act related question  if id 3 income tax related question an d id 4 the function forward into faq function
"""
@app.post("/bot")
async def create_item2(request: Request):
    b_json = await request.body()
    dt = json.loads(b_json)
    print(dt)
    txt_user = preprocess_text(dt['txt'], preprocess_functions)
    if dt['id']==1:
        re = "The following questions are related to GST, please select one of the following question to find answer"
        op = random.choices(population=data2['GST'], k=6)
    if dt['id']==2:
        re = "The following questions are related to Companies ACT, please select one of the following question to find answer"
        op = random.choices(population=data2['company_act'], k=6)
    if dt['id']==3:
        re = "The following questions are related to INCOME TAX, please select one of the following question to find answer"
        op = random.choices(population=data2['INCOME_TAX'], k=6)
    if dt['id']==4:
        if wordnet(txt_user) != None:
            if wordnet(txt_user)['score'] > 0.55:
                re = gentral_question_response(txt_user)
                op = random.choices(population=data2[wordnet(txt_user)['label']], k=6)

        else:
            re = "Sorry i wouldn't understand your query please choose below any one of the option"
            op =random.choices(population=q, k=6)
    return {"message":'sucess',"response":re,"options":list(set(op)),"component":"btn"}

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",port=5002,debug="True")
