import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from . import constants
import math




class WordEmbeddingGenerator(nn.Module):

    def __init__(self, word_to_idx, embedding_dim=64, dropout=constants.EMBEDDING_DROPOUT):
        '''
        :param word_to_idx: Word indices in the dictionary
        :param embedding_dim: Any chosen embedding dimension
        :param use_cuda:
        '''

        super(WordEmbeddingGenerator, self).__init__()
        self.word_to_idx = word_to_idx
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(len(self.word_to_idx), self.embedding_dim,
                                            padding_idx=0, scale_grad_by_freq=True)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, inp, drop=True):
        """
        :param sentence A list of strings, the text of the sentence
        :return A variable of shape (minibatch, embedding_dim, seq_len)
        """
        # TODO- Add position embeddings
        embeds = self.word_embeddings(inp)
        if drop:
            embeds = self.dropout(embeds)

        return embeds




class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        """
        :param x:
        :return:
        """
        A, B = x.chunk(2, dim=1)
        B_ = F.sigmoid(B)
        C =  torch.mul(A, B_)
        return C



class Attention(nn.Module):
    def __init__(self):
        """
        :param input_size:
        """
        super(Attention, self).__init__()

        self.softmax = nn.Softmax()

    def forward(self, decoder_rep, encoder_out):
        """
        :param decoder_state: Decoder state of Shape: (max_seq_len, embedding_dim)
        :param encoder_out: Encoder output of shape (max_seq_len, embedding_dim)
        :return: (2D tensor) Attention scores of shape (seq_len, seq_len)
        """

        attn = torch.mm(decoder_rep, encoder_out.t())
        #TODO- Check if only diagonal has to be considered
        #Basically, we need the dot product of each row in decoder_state and encoder_out
        attn_scores = self.softmax(attn)
        return attn_scores




class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_width=5, pad=0, dropout=constants.ENCODER_DROPOUT):
        """
        :param input_size:
        :param output_size:
        :param kernel_width:
        :param pad:
        """
        super(EncoderBlock, self).__init__()

        self.num_in_ch = input_size
        self.num_out_ch = output_size
        self.kwidth = kernel_width
        self.pad = pad
        self.conv = nn.Conv1d(self.num_in_ch, self.num_out_ch, self.kwidth, padding=pad)
        #see https://stackoverflow.com/questions/44212831/convolutional-nn-for-text-input-in-pytorch
        self.glu = GLU()
        self.dropout = nn.Dropout(p=dropout)


        for m in self.modules(): #TODO- THis is not what is described in paper. Change
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x, drop=True):
        """
        :param x: (3D tensor) Input of size (minibatch, in_ch, in_width).
                  NOTE: Here, in_ch is the embedding_dim
        :return: (3d tensor) Output of size (minibatch, out_ch, out_width)
                  NOTE: Here, out_ch is 2*embedding_dim and out_width is determined by the convolution
                  If a padding of (kernel_size-1)/2 is used, out_width is same as in_width
        """

        conv_out = self.conv(x)
        if drop:
            conv_out = self.dropout(conv_out)
        try:
            out = F.glu(conv_out, dim=1)
        except:
            out = self.glu.forward(conv_out)

        return out




class EncoderStack(nn.Module):
    def __init__(self, input_size, output_size, kernel_width=5, pad=0, num_layers=1, dropout=0.5):
        """
        :param input_size:
        :param output_size:
        :param kernel_width:
        :param pad:
        :param num_layers:
        """
        super(EncoderStack, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.encoders = nn.ModuleList([EncoderBlock(input_size, output_size, kernel_width, pad)
                                      for _ in range(self.num_layers)])

    def grad_mod(self, grad_input, grad_output):
        pass


    def forward(self, x, drop=True):
        """
        :param x:
        :return:
        """
        encoder_out = torch.zeros(x.size())
        for E in self.encoders:
            encoder_out = E.forward(x)
            # if drop:
            #     encoder_out = self.dropout(encoder_out)
            x = torch.add(encoder_out, x)  # Adding residual connections
            h = E.register_backward_hook(lambda grad: grad / (2*constants.NUM_DECODER_LAYERS))

        return encoder_out





class DecoderBlock(nn.Module):

    def __init__(self, input_size, output_size, kernel_width=5, pad=0, dropout=constants.DECODER_DROPOUT):
        """
        :param input_size: (int) Width of the input embedding
        :param output_size:
        :param kernel_width: (int) Size of the kernel
        :param pad: (int) Padding on each side of the input
        """
        super(DecoderBlock, self).__init__()
        self.num_in_ch = input_size
        self.num_out_ch = output_size
        self.kwidth = kernel_width
        self.pad = pad
        self.glu = GLU()
        self.conv = nn.Conv1d(self.num_in_ch, self.num_out_ch, self.kwidth, padding=pad)

        self.dropout = nn.Dropout(p=dropout)


        for m in self.modules(): #TODO- THis is not what is described in paper. Change
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x, drop=True):
        """
        :param x:
        :return: (3d tensor) Output of size (minibatch, out_ch, out_width)
                  NOTE: Here, out_ch is 2*embedding_dim and out_width is determined by the convolution
        """
        conv_out = self.conv(x)
        conv_out = conv_out.narrow(2, 0, -self.pad)
        if drop:
            conv_out = self.dropout(conv_out)
        try:
            out = F.glu(conv_out, dim = 1)
        except:
            out = self.glu.forward(conv_out)
        return out





class DecoderStack(nn.Module):
    def __init__(self, input_size, output_size, kernel_width=5, pad=0, num_layers=1):
        """
        :param input_size:
        :param kernel_width:
        :param pad:
        :param num_layers:
        """
        super(DecoderStack, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.kwidth = kernel_width
        self.decoders = nn.ModuleList([DecoderBlock(input_size, output_size, kernel_width, pad)
                                       for _ in range(self.num_layers)])
        self.linear_op = nn.Linear(self.input_size, self.input_size)

        self.attention = Attention()

    def forward(self, decoder_input, prev_target,
                encoder_out, encoder_input, batchsize, predict=False):
        """
        :param decoder_input: Input to the decoder stack of size (batch_size, embedding_size, seq_len)
        :param prev_target:
        :param encoder_out:
        :param encoder_input:
        :param batchsize:
        :return: Output of the last decoder layer (discarding k-1 trailing elements)
                  of size (
        """
        out = Variable(torch.zeros(decoder_input.size()))

        # NOTE: The below squeeze functions are assuming that batch size is always 1.
        # if not, it has to be changed
        encoder_input = encoder_input.squeeze(0).t()
        encoder_out = encoder_out.squeeze(0).t()
        encoder_rep = torch.add(encoder_out, encoder_input)
        prev_target = prev_target.squeeze(0).t()
        scaling_factor = decoder_input.size()[1] * math.sqrt(1. / decoder_input.size()[1])
        #INFO: decoder_input.size()[1] is the length of the sequence being decoded

        for D in self.decoders:
            out = D.forward(decoder_input)
            lin_out = self.linear_op(out.squeeze(0).t())
            decoder_rep = torch.add(lin_out, prev_target)
            attn_scores = self.attention.forward(decoder_rep, encoder_out)

            temp = torch.mm(attn_scores, encoder_rep) #TODO- Check if math is right
            candidate = torch.sum(temp, dim=0)
            candidate = torch.mul(candidate, scaling_factor) #Scaling to counteract variance
            candidate = candidate.unsqueeze(0).unsqueeze(2)
            candidate = torch.add(out, candidate)
            decoder_input = decoder_input + candidate
        if not predict:
            out_ = out
            # out_ = out.narrow(2, 1, out.size()[2]) #To remove the first element (EOS)
        else:

            out_ = out.split(1, dim=2)[-1]  #If predicting, only the last word is needed
        return out_

    def generate(self, decoder_input, prev_target,
                encoder_out, encoder_input, batchsize):
        pass



class HiddenToProb(nn.Module):
    def __init__(self, input_size, output_size, dropout=constants.H2O_DROPOUT):
        super(HiddenToProb, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, state, drop=False):
        scores = self.linear(state)
        if drop:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores
