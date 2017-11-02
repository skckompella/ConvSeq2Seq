
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch
import copy
import os
import random

from . import layers
from . import constants

MAX_LEN = 1024
class ConvS2SAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        argparser.add_arg('-hs', '--embedding_size', type=int, default=constants.EMBEDDING_SIZE,
            help='size of the embeddings')
        argparser.add_arg('-nel', '--num_encoder_layers', type=int, default=constants.NUM_ENCODER_LAYERS,
            help='number of encoder layers')
        argparser.add_arg('-ndl', '--num_decoder_layers', type=int, default=constants.NUM_DECODER_LAYERS,
                          help='number of decoder layers')
        argparser.add_arg('-ks', '--kernel_size', type=int, default=constants.KERNEL_SIZE,
            help='size of the convolution kernel')
        argparser.add_arg('-lr', '--learning_rate', type=float, default=constants.LEARNING_RATE,
            help='learning rate')
        argparser.add_arg('-dr', '--dropout', type=float, default=0.1,
            help='dropout rate')
        argparser.add_arg('--cuda', action='store_true', default=constants.USE_CUDA,
            help='disable GPUs even if available')
        argparser.add_arg('--gpu', type=int, default=-1,
            help='which GPU device to use')


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.set_device(opt['gpu'])

        if not shared:
            self.dict = DictionaryAgent(opt)
            self.id = 'ConvS2S'
            self.EOS = self.dict.end_token
            self.SOS = self.dict.start_token
            self.use_cuda = opt['cuda']

            self.EOS_TENSOR = torch.LongTensor(self.dict.parse(self.EOS))
            self.SOS_TENSOR = torch.LongTensor(self.dict.parse(self.SOS))

            self.kernel_size = opt['kernel_size']
            self.embedding_size = opt['embedding_size']
            self.num_enc_layers = opt['num_encoder_layers']
            self.num_dec_layers = opt['num_decoder_layers']

            self.longest_label = 2
            self.encoder_pad = (self.kernel_size - 1) // 2
            self.decoder_pad = self.kernel_size - 1

            self.criterion = nn.NLLLoss()
            self.embeder = layers.WordEmbeddingGenerator(self.dict.tok2ind,
                                                         embedding_dim=self.embedding_size)
            self.encoder = layers.EncoderStack(self.embedding_size,
                                               2*self.embedding_size,
                                          self.kernel_size,
                                          self.encoder_pad,
                                          self.num_enc_layers)

            self.decoder = layers.DecoderStack(self.embedding_size,
                                               2 * self.embedding_size,
                                               self.kernel_size,
                                               self.decoder_pad,
                                               self.num_dec_layers)

            self.h2o = layers.HiddenToProb(self.embedding_size, len(self.dict))

            lr = opt['learning_rate']
            self.optims = {
                'embeds': optim.Adam(self.embeder.parameters(), lr=lr),
                'encoder': optim.Adam(self.encoder.parameters(), lr=lr),
                'decoder': optim.Adam(self.decoder.parameters(), lr=lr),
                'd2o': optim.Adam(self.h2o.parameters(), lr=lr),
            }
            if self.use_cuda:
                self.cuda()
            if 'model_file' in opt and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        self.episode_done = True


    def parse(self, text):
        if self.use_cuda:
            return torch.cuda.LongTensor(self.dict.txt2vec(text))
        else:
            return torch.LongTensor(self.dict.txt2vec(text))

    def v2t(self, vec):
        return self.dict.vec2txt(vec)

    def cuda(self):
        self.EOS_TENSOR = self.EOS_TENSOR.cuda(async=True)
        self.SOS_TENSOR = self.SOS_TENSOR.cuda(async=True)
        self.criterion.cuda()
        self.embeder.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        # self.attention.cuda()
        self.h2o.cuda()

    def zero_grads(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path
        model = {}
        model['embeds'] = self.embeder.state_dict()
        model['encoder'] = self.encoder.state_dict()
        model['decoder'] = self.decoder.state_dict()
        model['d2o'] = self.h2o.state_dict()
        model['longest_label'] = self.longest_label

        with open(path, 'wb') as write:
            torch.save(model, write)

    def load(self, path):
        with open(path, 'rb') as read:
            model = torch.load(read)

        self.embeder.load_state_dict(model['embeds'])
        self.encoder.load_state_dict(model['encoder'])
        self.decoder.load_state_dict(model['decoder'])
        self.h2o.load_state_dict(model['d2o'])
        self.longest_label = model['longest_label']

    def update(self, xs, ys):
        #NOTE: Batchsize is always 1. i.e. One turn.
        # Seq len is the number of words in that turn
        batchsize, seq_len = xs.size()
        encoder_input = self.embeder.forward(xs).permute(0,2,1)

        # first encode context
        encoder_out = self.encoder.forward(encoder_input)
        if len(encoder_out.size()) == 2: #If there is only one word in the input
            encoder_out.unsqueeze(2)

        targets_embedded = self.embeder.forward(ys).permute(0,2,1)
        # start with EOS tensor for all
        x = Variable(self.EOS_TENSOR)
        xe = self.embeder.forward(x).unsqueeze(2)
        decoder_input = targets_embedded
        prev_target = torch.cat((xe, decoder_input), 2)
        prev_target = prev_target.narrow(2, 0, -1) #Removing the last target

        output_lines = [[] for _ in range(batchsize)]
        self.zero_grads()

        # update model
        loss = 0
        self.longest_label = max(self.longest_label, ys.size(1))

        out = self.decoder.forward(decoder_input, prev_target,
                                   encoder_out, encoder_input, batchsize)

        #NOTE: For the linear layer below and loss calculations,
        #      a 'batch' is the number of words in the sentence.
        #This is a hack job. Need to fix #FIXME
        preds, scores = self.h2o.forward(out.squeeze(dim=0).t())
        loss += self.criterion(scores, ys.squeeze())

        for i in range(batchsize):
            for j in range(preds.size(0)):
                token = self.v2t([preds.data[j]])
                output_lines[i].append(token)

        loss.backward()
        self.update_params()

        if random.random() < 0.1:
            true = self.v2t(ys.data[0])
            print('loss:', round(loss.data[0], 2),
                 ' '.join(output_lines[0]), '(true: {})'.format(true))

        return output_lines

    def predict(self, xs):
        batchsize, seq_len = xs.size()
        encoder_input = self.embeder.forward(xs)
        encoder_input = encoder_input.permute(0,2,1)

        # first encode context
        encoder_out = self.encoder.forward(encoder_input)
        if len(encoder_out.size()) == 2: #If there is only one word in the input
            encoder_out.unsqueeze(2)

        # start with EOS tensor for all
        x = Variable(self.EOS_TENSOR)
        if self.use_cuda:
            x = x.cuda(async=True)
        xe = self.embeder.forward(x).unsqueeze(2)
        decoder_input = xe
        prev_target = Variable(torch.zeros(decoder_input.size()))
        if self.use_cuda:
            prev_target = prev_target.cuda(async=True)
        # prev_target = xe
        output_lines = [[] for _ in range(batchsize)]
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0
        token_count = 0

        while (total_done < batchsize) and max_len < self.longest_label:
            out = self.decoder.forward(decoder_input, prev_target,
                                       encoder_out, encoder_input, batchsize,
                                       predict=True)
            preds, scores = self.h2o.forward(out.squeeze(dim=0).t())
            prev_target = self.embeder.forward(preds).unsqueeze(2)

            decoder_input = torch.cat((decoder_input, prev_target), dim=2)
            token_count += 1
            if token_count > 1: #To ignore the first generated string
                max_len += 1

                for i in range(batchsize):
                    eos_count = 0
                    if not done[i]:
                        token = self.v2t(preds.data)
                        # print('eos_count' , eos_count)
                        if token == self.EOS:
                            done[i] = True
                            total_done += 1
                            token_count = 0
                        else:
                            output_lines[i].append(token)
                            # eos_count+=1

        # if random.random() < 0.1:
        print('prediction:', ' '.join(output_lines[0]))
        return output_lines


    def batchify(self, obs):
        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]
        batchsize = len(exs)

        parsed = [self.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed])
        if self.use_cuda:
            xs = torch.cuda.LongTensor(batchsize, max_x_len).fill_(0)
        else:
            xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        for i, x in enumerate(parsed):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        if self.use_cuda:
            xs = xs.cuda(async=True)
        xs = Variable(xs)
        ys = None
        if 'labels' in exs[0]:
            labels = [random.choice(ex['labels']) + ' ' + self.EOS for ex in exs]
            parsed = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            if self.use_cuda:
                ys = torch.cuda.LongTensor(batchsize, max_y_len).fill_(0)

            else:
                ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                ys = ys.cuda(async=True)
            ys = Variable(ys)
        return xs, ys, valid_inds


    def batch_act(self, observations):
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, valid_inds = self.batchify(observations)
        if len(xs) == 0:
            return batch_reply

        # Either train or predict
        if ys is not None:
            # predictions = self.predict(xs)

            predictions = self.update(xs, ys)
        else:
            predictions = self.predict(xs)

        for i in range(len(predictions)):
            batch_reply[valid_inds[i]]['text'] = ' '.join(
                c for c in predictions[i] if c != self.EOS)

        return batch_reply


    def act(self):
        return self.batch_act([self.observation])[0]


    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation


