import torch
import torch.nn as nn
# from torch.autograd import Variable
import utils
import models
import random


class seq2seq(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(seq2seq, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.rnn_encoder(config)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.rnn_decoder(config, embedding=tgt_embedding, use_attention=use_attention)
        self.conv_decoder = models.conv_decoder(config, embedding=self.decoder.embedding, use_attention=use_attention)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
        #self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduce=False)
        if config.use_cuda:
            self.criterion.cuda()
        self.l1loss = nn.SmoothL1Loss()

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss

    def forward(self, src, src_len, dec, targets, teacher_ratio=1.0):
        src = src.t()
        dec = dec.t()
        targets = targets.t()
        teacher = random.random() < teacher_ratio

        #contexts, state, embeds = self.encoder(src, src_len.data.tolist())
        contexts, state = self.encoder(src, src_len.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)

        outputs = []

        if teacher:
            conv_outputs, conv_scores = self.conv_decoder(state)
            convs = conv_outputs
            for input in dec.split(1):
                output, state, attn_weights, weights_conv = self.decoder(input.squeeze(0), state, convs)
                outputs.append(output)
            outputs = torch.stack(outputs)
            #print(outputs.size())
        else:
            inputs = [dec.split(1)[0].squeeze(0)]
            conv_outputs, conv_scores = self.conv_decoder(state)
            convs = conv_outputs
            for i, _ in enumerate(dec.split(1)):
                output, state, attn_weights, weights_conv = self.decoder(inputs[i], state, convs)
                predicted = output.max(1)[1]
                inputs += [predicted]
                outputs.append(output)
            outputs = torch.stack(outputs)

        tar = targets[:28, :]
        tar_emb = self.decoder.embedding(tar)
        # tar_emb = Variable(tar_emb.data, requires_grad=False, volatile=False)
        tar_emb = torch.tensor(tar_emb, requires_grad=False).detach()
        conv_scores = conv_scores.transpose(0,1)
        # print(outputs.size(), conv_scores.size(), targets.size(), tar_emb.size())
        loss = self.compute_loss(outputs, targets).mean() + self.l1loss(conv_outputs, tar_emb.transpose(0,1)) \
               + self.compute_loss(conv_scores.contiguous(), tar).mean()
        return loss, outputs

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        #bos = Variable(torch.ones(src.size(0)).long().fill_(utils.BOS), volatile=True)
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        #contexts, state, embeds = self.encoder(src, lengths.data.tolist())
        contexts, state = self.encoder(src, lengths.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        conv_outputs, conv_scores = self.conv_decoder(state)
        convs = conv_outputs
        for i in range(self.config.max_time_step):
            output, state, attn_weights, weights_conv = self.decoder(inputs[i], state, convs)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        #sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t().data
        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t()

        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            #alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t().data
            alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t()
        else:
            alignments = None

        return sample_ids, alignments

    def beam_sample(self, src, src_len, beam_size=1, eval_=False):

        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        # contexts, encState, embeds = self.encoder(src, lengths.data.tolist())
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            # return Variable(a, volatile=True)
            return torch.tensor(a, requires_grad=False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.

        # contexts = rvar(contexts.data)
        contexts = rvar(contexts)
        # embeds = rvar(embeds.data)

        if self.config.cell == 'lstm':
            # decState = (rvar(encState[0].data), rvar(encState[1].data))
            decState = (rvar(encState[0]), rvar(encState[1]))
        else:
            # decState = rvar(encState.data)
            decState = rvar(encState)
        #print(decState[0].size(), memory.size())
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        conv_outputs, conv_scores = self.conv_decoder(decState)
        convs = conv_outputs

        for i in range(self.config.max_time_step):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn, weights_conv = self.decoder(inp, decState, convs)
            # decOut: beam x rnn_size
            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                # b.advance(output.data[:, j], attn.data[:, j])
                b.advance(output[:, j], attn[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        # for j in ind.data:
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])

        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn
