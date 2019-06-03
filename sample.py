import torch


class Sample:
    def __init__(self, sentence, text_id, sent_id):
        """
        Create a sample with a source sentence and label.
        :param sentence: string
        :param label_seq: target sequence corresponding to the string.
        :text_id: string indicating the text identifier.
        :sent_id: int indicating the sentence number in the corpus.
        """
        super().__init__()
        self.sentence = [x.replace("m_", "").replace("l_", "") for x in sentence.lower().split()]
        self.label_seq = [ 1 if "M_" in x else ( 0 if "L_" in x else -1 ) for x in sentence.split()]
        assert(len(self.label_seq) == len(self.sentence))
        self.text_id = text_id
        self.sent_id = int(sent_id)

        # To be initialised after collecting discourse
        self.discourse = []
        self.focus_position = 0
        self.discourse_length = 0
        self.max_length = 0

    def update_discourse(self, discourse, focus_position):
        self.discourse = discourse
        self.focus_position = focus_position
        self.discourse_length = len([x for x in self.discourse if x != []])
        self.max_length = max([len(x) for x in self.discourse])

class Batch:
    def __init__(self):
        # Focus sentence
        self.words = []
        self.idx = []
        self.lengths = []
        self.discourse_lengths = []
        self.focus_positions = []

        # Labels of the focus sentence
        self.label_seqs = []

    def to_cuda(self):
        if torch.cuda.is_available():
            self.idx = self.idx.cuda()
            self.lengths = self.lengths.cuda()
            self.discourse_lengths = self.discourse_lengths.cuda()
            self.focus_positions = self.focus_positions.cuda()

            # Labels to cuda
            self.label_seqs = self.label_seqs.cuda()

    def to_tensor(self):
        self.idx = torch.LongTensor(self.idx)
        self.lengths = torch.LongTensor(self.lengths)
        self.discourse_lengths = torch.LongTensor(self.discourse_lengths)
        self.focus_positions = torch.LongTensor(self.focus_positions)

        # Labels to tensor
        try:
            self.label_seqs = torch.LongTensor(self.label_seqs)
        except:
            print(self.label_seqs)
