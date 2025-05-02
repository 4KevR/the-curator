import argparse
import logging
import requests
import json
import time
import base64
import sys
from threading import Thread
from sseclient import SSEClient
from queue import Queue


class remote_ASRmodel:
    def __init__(self, args):
        self.args = args
        self.text_queue = Queue()
        self.audio_source = self.get_audio_input()
    
    def get_audio_input(self):
        args = self.args
        if args.input == 'portaudio':
            from asr.pythonRecordingClient.pyaudioStreamAdapter import PortaudioStream
            logging.debug("Using portaudio as input. If you want to use ffmpeg specify '-i ffmpeg'.")
            stream_adapter = PortaudioStream()
            input = args.audiodevice
            if args.list:
                stream_adapter.print_all_devices()
            if args.audiodevice < 0:
                logging.debug("The portaudio backend requires the '-a' parameter. Run python client.py -L to see the available audio devices.")
                exit(1)
        else:
            from asr.pythonRecordingClient._helper import BugException
            raise BugException()

        stream_adapter.set_input(input)

        return stream_adapter
    

    def run_session(self):
        args = self.args
        sessionID, streamID = self.set_graph()

        t = Thread(target=self.read_text,
                args=(args.url, sessionID, args.api, args.token))
        t.daemon = True
        t.start()

        time.sleep(1) # To make sure the SSEClient is running before sending the INFORMATION request

        logging.debug("Requesting worker informations")
        data={'controll':"INFORMATION"}
        info = requests.post(args.url + "/"+args.api+"/" + sessionID + "/" + streamID + "/append", 
                             json=json.dumps(data), cookies={'_forward_auth': args.token})
        if info.status_code != 200:
            logging.debug("ERROR in requesting worker information")
            sys.exit(1)

        self.send_session(args.url, sessionID, streamID, self.audio_source, args.api, args.token)
        t.join()


    def set_graph(self):
        args = self.args
        logging.debug("Requesting default graph for ASR")
        d={}
        res = requests.post(args.url + "/"+args.api+"/get_default_asr", 
                            json=json.dumps(d), cookies={'_forward_auth': args.token})
        if res.status_code != 200:
            if res.status_code == 401:
                logging.debug("You are not authorized. " \
                "Either authenticate with --url https://$username:$password@$server or with --token $token where you get the token from "+args.url+"/gettoken")
            else:
                logging.debug("Status: {}, Text: {}".format(res.status_code,res.text))
                logging.debug("ERROR in requesting default graph for ASR")
            sys.exit(1)
        sessionID, streamID = res.text.split()

        logging.debug("Setting properties")
        graph=json.loads(requests.post(args.url+"/"+args.api+"/"+sessionID+"/getgraph", cookies={'_forward_auth': args.token}).text)
        logging.debug("Graph: {}".format(graph))

        return sessionID, streamID


    def send_session(self, url, sessionID, streamID, audio_source, api, token):
        try:
            self.send_start(url, sessionID, streamID, api, token)
            while (True):
                self.send_audio(audio_source, url, sessionID, streamID, api, token)

        except KeyboardInterrupt:
            logging.debug("Caught KeyboardInterrupt")

        time.sleep(1)
        self.send_end(url, sessionID, streamID, api, token)


    def send_start(self, url, sessionID, streamID, api, token):
        logging.debug("Start sending audio")
        
        data={'controll':"START"}
        info = requests.post(url + "/"+api+"/" + sessionID + "/" + streamID + "/append", json=json.dumps(data), cookies={'_forward_auth': token})
        
        if info.status_code != 200:
            logging.debug("ERROR in starting session")
            sys.exit(1)


    def send_audio(self, audio_source, url, sessionID, streamID, api, token, raise_interrupt=True):
        chunk = audio_source.read()
        chunk = audio_source.chunk_modify(chunk)
        s = time.time()
        e = s + len(chunk)/32000
        data = {"b64_enc_pcm_s16le":base64.b64encode(chunk).decode("ascii"),"start":s,"end":e}
        res = requests.post(url + "/"+api+"/" + sessionID + "/" + streamID + "/append", json=json.dumps(data), cookies={'_forward_auth': token})
        if res.status_code != 200:
            logging.debug("ERROR in sending audio")
            sys.exit(1)
        #else:
            #logging.debug(len(chunk))


    def send_end(self, url, sessionID, streamID, api, token):
        logging.debug("Sending END.")
        data={'controll': "END"}
        res = requests.post(url + "/"+api+"/" + sessionID + "/" + streamID + "/append", json=json.dumps(data), cookies={'_forward_auth': token})
        if res.status_code != 200:
            logging.debug("ERROR in sending END message")
            sys.exit(1)


    def read_text(self, url, sessionID, api, token):
        logging.debug("Starting SSEClient")
        messages = SSEClient(url + "/"+api+"/stream?channel=" + sessionID)

        for msg in messages:
            if len(msg.data) == 0:
                break
            try:
                data = json.loads(msg.data)
                if "seq" in data:
                    self.text_queue.put(data["seq"])

            except json.decoder.JSONDecodeError:
                logging.debug("WARNING: json.decoder.JSONDecodeError (this may happend when running tts system but no video generation)")
                continue


    def text_consumer(self):
        while True:
            sentence = asr.text_queue.get()
            print(sentence)




def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-u", "--url", 
                        default="", 
                        help="Where to send the audio to")
    
    parser.add_argument('--token', 
                        default=None, 
                        help='Webapi access token for authentication')
    
    parser.add_argument('-i', '--input', 
                        default='portaudio', 
                        choices=['portaudio', 'ffmpeg','link'],
                        help="Which input type should be used")
    
    """PyAudio/Portaudio"""
    parser.add_argument('-L', '--list', 
                        action='store_true',
                        help='Pyaudio. List audio available audio devices')
    parser.add_argument('-a', '--audiodevice', 
                        default=1, type=int, 
                        help='Pyaudio. Index of audio device to use')
    parser.add_argument('-ch', '--audiochannel', 
                        type=int, default=None, 
                        help='index of audio channel to use (first channel = 1)')

    """ Properties """
    parser.add_argument('--llm', 
                        default="", 
                        help="URL to query LLM using text generation client")
    
    args = parser.parse_args()

    args.api = "webapi"

    return args




if __name__ == "__main__":
    args = parseArgs()
    args.token = ""
    print("args:", args)

    asr = remote_ASRmodel(args)
    Thread(target=asr.text_consumer, args=(asr,), daemon=True).start()
    asr.run_session()
