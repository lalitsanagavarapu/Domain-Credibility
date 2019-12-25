from dotenv import load_dotenv
from flask import render_template
from flask import request
from utils.databases import Features
from utils.essentials import app
from utils.essentials import Database
from utils.essentials import WebcredError
from utils.webcred import Webcred

import json
import logging
import os
import requests
import subprocess
import time


load_dotenv(dotenv_path='.env')
logger = logging.getLogger('WEBCred.app')
logging.basicConfig(
    filename='log/logging.log',
    filemode='a',
    format='[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)


class Captcha(object):
    def __init__(self, resp=None, ip=None):
        google_api = 'https://www.google.com/recaptcha/api/siteverify'
        self.url = google_api
        self.key = '6LcsiCoUAAAAAL9TssWVBE0DBwA7pXPNklXU42Rk'
        self.resp = resp
        self.ip = ip
        self.params = {
            'secret': self.key,
            'response': self.resp,
            'remoteip': self.ip
        }

    def check(self):
        result = requests.post(url=self.url, params=self.params).text
        result = json.loads(result)
        return result.get('success', None)


@app.route("/start", methods=['GET'])
def start():

    addr = request.environ.get('REMOTE_ADDR')
    g_recaptcha_response = request.args.get('g-recaptcha-response', None)
    response_captcha = Captcha(ip=addr, resp=g_recaptcha_response)

    if not response_captcha.check():
        pass
        # result = "Robot not allowed"
        # return result

    data = collectData(request)
    del data['_sa_instance_state']# If not deleted gives json object not serializable error.
    #print("Part4",data)

     
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


def collectData(request):

    try:

        database = Database(Features)
        dt = Webcred(database, request)
        data = dt.assess()
        #print("Part3",data)

    except WebcredError as e:
        #data['Error'] = {e.message}
        print("JSON CONVERT",str(e),e.message)
        data['Error']=str(e)

    # logger.info(data)
    return data


def appinfo(url=None):
    pid = os.getpid()
    # print pid
    cmd = ['ps', '-p', str(pid), '-o', "%cpu,%mem,cmd"]
    # print
    while True:
        info = subprocess.check_output(cmd)
        print(info)
        time.sleep(3)

    print('exiting appinfo')
    return None


if __name__ == "__main__":
    # TODO: start standford-server from here
    '''
    cd stanford-corenlp-full-2018-02-27;
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
    -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port
    9000 -timeout 30000 --add-modules java.se.ee`
    '''
    app.run(
        threaded=True,
        host='0.0.0.0',
        debug=True,
        port=8000,
    )
