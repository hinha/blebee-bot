import json
import os
import random

import nltk
import tflearn
import pickle
import numpy as np
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

dir_path = os.path.dirname(os.path.realpath("blebee-bot"))


class TrainBot(object):
    ERROR_THRESHOLD = 0.25

    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        self.intents = None
        with open(f"{dir_path}/intent.json") as f:
            self.intents = json.load(f)

        self.docs = []
        self.classes = []
        self.words = []

        self.load_model = os.path.join(dir_path + "/training.pickle")

    def _preprocessing(self, intents: dict):

        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                # tokenize setiap kata didalam kalimat
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)

                self.docs.append((w, intent["tag"]))

                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        ignore_words = ["?"]
        words = [self.stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
        self.words = sorted(list(set(words)))

    def training(self, debug=False):
        try:
            with open(self.load_model, "rb") as f:
                self.words, self.classes, training, output = pickle.load(f)
        except FileNotFoundError as e:
            self._preprocessing(self.intents)

            training = []
            output = []
            output_empty = [0] * len(self.classes)

            for doc in self.docs:
                bag = []
                pattern_words = doc[0]
                pattern_words = [self.stemmer.stem(w.lower()) for w in pattern_words]

                for w in self.words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                # output is a '0' for each tag and '1' for current tag
                output_row = list(output_empty)
                output_row[self.classes.index(doc[1])] = 1
                training.append([bag, output_row])
                output.append(output_row)

            random.shuffle(training)
            training = np.array(training)
            output = np.array(output)

            with open("training.pickle", "wb") as f:
                pickle.dump((self.words, self.classes, training, output), f)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        tf.reset_default_graph()

        input_h = tflearn.input_data(shape=(None, len(train_x[0])), dtype=tf.float32)
        h2 = tflearn.fully_connected(input_h, 9)
        h3 = tflearn.fully_connected(h2, 18)
        h4 = tflearn.fully_connected(h3, 18)

        h5 = tflearn.fully_connected(h4, 9)
        output_h = tflearn.fully_connected(h5, len(train_y[0]), activation='softmax')
        output_h_reg = tflearn.regression(output_h)

        model = tflearn.DNN(output_h_reg, tensorboard_dir='tflearn_logs')

        if debug:
            model.fit(train_x, train_y, n_epoch=5000, batch_size=10, show_metric=True)
            model.save('model/model.tflearn')
        else:
            model.load("model/model.tflearn")

        return model

    # bag of word
    def bow(self, sentence, words, show_details=False):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for sw in sentence_words:
            for i, v in enumerate(words):
                if v == sw:
                    bag[i] = 1
                    if show_details:
                        print(f"test : {v}")

        return np.array(bag)

    def _clean_up_sentence(self, sentence) -> list:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def classify(self, sentence, model_output: tflearn.DNN, user_id="default") -> list:
        results = model_output.predict([self.bow(sentence, self.words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))

        # array is bad leaning
        msgError = []
        for n in return_list:
            err = np.array(n[1])
            if err < 0.50:
                msgError.append(err)

        # return length bad learning is greater than 2
        if len(msgError) > 1:
            msgError.sort(reverse=True)
            return ['Noel masih belajar nih, bisa diperjelas lagi pertanyaan nya?', 'error', '',
                    msgError[0].tolist()]

        context = {}
        for i in self.intents["intents"]:
            if i["tag"] == return_list[0][0]:
                context[user_id] = i["context_set"]

                output_var = [random.choice(i['responses']), i['tag'], context[user_id]]
                acc = return_list[0][1].tolist()
                output_var.append(acc)
                return output_var

            # if 'context_filter' not in i or \
            #         (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
            #     context_filter = i.get('context_filter', '')
            #     output_var = [random.choice(i['responses']), i['tag'], context_filter]
            #     return output_var


if __name__ == '__main__':
    # TEXT = "twitter-keyword 2020-01-30:@skoiahdqqdh"
    test = TrainBot()
    # model = test.training()

    # os.remove(os.path.join(dir_path + "/training.pickle"))
    # test.training(debug=True)
