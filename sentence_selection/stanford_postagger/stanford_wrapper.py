# this script requires MaxentTaggerServer to be running on port 9000
# change to this directory
# command: java -cp stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTaggerServer -model models/english-left3words-distsim.tagger -port 9000

import socket

class StanfordPOSTagger():
    def tag(self, sentence):
        buffer_size = (1 << 10)
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.connect(("localhost", 9000))
            connection.sendall((sentence + "\n").encode('utf-8'))
            data = b""
            while True:
                received = connection.recv(buffer_size)
                if not received: break
                data += received
            return self.parse_output(data.decode())
        except Exception as e:
            print(e)
            return []
        finally:
            connection.close()


    def parse_output(self, text):
      tagged_sentences = []
      for tagged_sentence in text.strip().split(" "):
          tagged_word = tagged_sentence.strip().split("_")
          tagged_sentences.append(tuple((tagged_word[0], tagged_word[1])))
      return tagged_sentences
