import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageInputUnit(nn.Module):
    """
    Input: Tensor of shape (N, image_input.dim, H, W)
    Output: Tensor of shape (N, H*W, memory_dim)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        # self.conv2 = nn.Conv3d(out_channels, out_channels, 1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0))
        self.out_channels = out_channels

    def forward(self, images):
        batch_size = images.size(0)
        x = F.elu(self.conv1(images))
        x = F.elu(self.conv2(x))
        return x.view(batch_size, self.out_channels, -1).transpose(1, 2)


class QuestionInputUnit(nn.Module):
    """
    Input: questions, question_lengths
        questions: Padded sequence of word index, Tensor of shape (T, N)
        question_lengths: LongTensor of shape (N)
    Output: question_representation, contextual_words
        question_representation: Tensor of shape (N, control_dim)
        contextual_words: Tensor of shape (N, T, control_dim)
    """

    def __init__(self, n_vocab, embedding_dim=300, control_dim=512, glove_path=None):
        super().__init__()

        if glove_path is not None:
            # logger.info('Using GloVe')
            print('Using GloVe')
            self.embedding = nn.Embedding.from_pretrained(torch.load(glove_path))
        else:
            self.embedding = nn.Embedding(n_vocab, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, control_dim // 2, bidirectional=True)

    def forward(self, questions, question_lengths):

        total_length = questions.size(1)
        questions = self.embedding(questions)
        questions = nn.utils.rnn.pack_padded_sequence(questions, question_lengths, batch_first=True)

        # self.lstm.flatten_parameters()
        output, (h_n, _) = self.lstm(questions)
        contextual_words, _lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                                      total_length=total_length)

        question_representation = torch.cat((h_n[0], h_n[1]), dim=-1)
        return question_representation, contextual_words


class ControlUnit(nn.Module):
    """
    Input: position_aware_question, contextual_words, previous_control
        position_aware_question: Tensor of shape (N, control_dim)
        contextual_words: Tensor of shape (N, T, control_dim)
        previous_control: Tensor of shape (N, control_dim)
    Output: Tensor of shape (N, control_dim)
    """

    def __init__(self, control_dim: int):
        super().__init__()

        self.merge_input_1 = nn.Linear(control_dim * 2, control_dim)
        self.merge_input_2 = nn.Linear(control_dim, control_dim)

        self.logits = nn.Linear(control_dim, 1)

    def forward(self, position_aware_question, contextual_words, previous_control):
        cq = torch.cat((previous_control, position_aware_question), dim=-1)
        cq = torch.tanh(self.merge_input_1(cq))
        cq = self.merge_input_2(cq)

        interactions = cq.unsqueeze(1) * contextual_words
        logits = self.logits(interactions)
        attention = F.softmax(logits, dim=1)
        next_control = (contextual_words * attention).sum(dim=1)
        return next_control


class ReadUnit(nn.Module):
    """
    Input: knowledge_base, control, previous_memory
        knowledge_base: Tensor of shape (N, S, memory_dim)
        control: Tensor of shape (N, control_dim)
        previous_memory: Tensor of shape (N, memory_dim)
    Output: Tensor of shape (N, memory_dim)
    """

    def __init__(self, memory_dim: int):
        super().__init__()

        self.knowledge_base_proj = nn.Linear(memory_dim, memory_dim)
        self.previous_memory_proj = nn.Linear(memory_dim, memory_dim)
        self.cat_proj = nn.Linear(memory_dim * 2, memory_dim)

        self.logits = nn.Linear(memory_dim, 1)

    def forward(self, knowledge_base, control, previous_memory):
        knowledge_base_projected = self.knowledge_base_proj(knowledge_base)
        previous_memory_projected = self.previous_memory_proj(previous_memory)

        interaction = previous_memory_projected.unsqueeze(1) * knowledge_base_projected
        interaction = torch.cat((interaction, knowledge_base), dim=-1)
        interaction = self.cat_proj(interaction)
        interaction = F.elu(interaction)

        interaction = interaction * control.unsqueeze(1)
        interaction = F.elu(interaction)

        logits = self.logits(interaction)
        attention = F.softmax(logits, dim=1)
        info = (attention * knowledge_base).sum(dim=1)
        return info


class WriteUnit(nn.Module):
    """
    Input: info, previous_memory
        info: Tensor of shape (N, memory_dim)
        previous_memory: Tensor of shape (N, memory_dim)
    Output: Tensor of shape (N, memory_dim)
    """

    def __init__(self, memory_dim: int, self_attention=False, memory_gate=False):
        super().__init__()

        if self_attention:
            raise NotImplementedError()
        if memory_gate:
            raise NotImplementedError()

        self.write_memory_proj = nn.Linear(2 * memory_dim, memory_dim)

    def forward(self, info, previous_memory):
        next_memory = torch.cat((info, previous_memory), dim=-1)
        next_memory = self.write_memory_proj(next_memory)
        return next_memory


class MACCellState:
    def __init__(self, control, memory):
        self.control = control
        self.memory = memory


class MACCell(nn.Module):
    def __init__(self, dim: int, net_length=12, self_attention=False, memory_gate=False, dropout=0.15):
        super().__init__()

        control_dim, memory_dim = dim, dim

        self.control = ControlUnit(control_dim)
        self.read = ReadUnit(memory_dim)
        self.write = WriteUnit(memory_dim)

    def forward(self, knowledge_base, position_aware_question, contextual_words, previous_state: MACCellState):
        control = self.control(position_aware_question, contextual_words, previous_state.control)
        info = self.read(knowledge_base, control, previous_state.memory)
        memory = self.write(info, previous_state.memory)

        return MACCellState(control, memory)


class MAC(nn.Module):
    def __init__(self, dim=512, net_length=12, self_attention=False, memory_gate=False, dropout=0.15):
        super().__init__()

        self.dim = dim
        self.net_length = net_length

        self.cell = MACCell(dim, net_length=net_length, self_attention=self_attention, memory_gate=memory_gate,
                            dropout=dropout)

        self.init_control = nn.Parameter(torch.empty(dim))
        self.init_memory = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.init_control)
        nn.init.normal_(self.init_memory)

        self.question_input_proj = nn.Linear(dim, dim)
        self.question_position_transform = nn.ModuleList(
            [nn.Linear(dim, dim) for i in range(self.net_length)])

    def forward(self, knowledge_base, question_representation, contextual_words):
        batch_size = knowledge_base.size(0)
        state = MACCellState(self.init_control.expand(batch_size, -1), self.init_memory.expand(batch_size, -1))

        question_representation = self.question_input_proj(question_representation)
        question_representation = torch.tanh(question_representation)

        for i in range(self.net_length):
            position_aware_question = self.question_position_transform[i](question_representation)
            state = self.cell(knowledge_base, position_aware_question, contextual_words, state)
        return state


class Classifier(nn.Module):
    """
    Input:
        memeory: Tensor of shape (N, memory_dim)
        question_representation: Tensor of shape (N, control_dim)
    Output: Tensor of shape (N, classes)
    """

    def __init__(self, dim, classes):
        super().__init__()

        memory_dim, control_dim = dim, dim

        self.linear_question = nn.Linear(control_dim, memory_dim)
        self.fc1 = nn.Linear(2 * memory_dim, memory_dim)
        self.fc2 = nn.Linear(memory_dim, classes)

    def forward(self, memory, question_representation):
        q2 = self.linear_question(question_representation)
        features = torch.cat((memory, q2), dim=-1)
        out = F.elu(self.fc1(features))
        out = self.fc2(out)
        return out


class MACNetwork(nn.Module):
    def __init__(self, n_vocab, classes, in_channels=1024, dim=512, net_length=12, embedding_dim=300,
                 self_attention=False,
                 memory_gate=False, dropout=0.15, glove_path=None):
        super().__init__()

        self.image_input = ImageInputUnit(in_channels, dim)
        self.question_input = QuestionInputUnit(n_vocab, embedding_dim=embedding_dim, control_dim=dim,
                                                glove_path=glove_path)
        self.mac = MAC(dim, net_length=net_length, self_attention=self_attention, memory_gate=memory_gate,
                       dropout=dropout)
        self.classifier = Classifier(dim, classes)

    def forward(self, images, questions, question_lengths):
        knowledge_base = self.image_input(images)
        question_representation, contextual_words = self.question_input(questions, question_lengths)
        final_state = self.mac(knowledge_base, question_representation, contextual_words)
        return self.classifier(final_state.memory, question_representation)


if __name__ == '__main__':
    model = MACNetwork(10, 1)

    v = torch.rand(2, 1024, 14, 14)
    q = torch.randint(0, 10, (10, 2))

    q_len = [10, 10]

    y = model(v, q, q_len)

    print(y)
