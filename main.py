import numpy as np
import copy
import random
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch import jit

from typing import Dict, Any, Optional, Tuple, List, NamedTuple, Generator, cast

from tqdm import tqdm

from pathlib import Path

from transformers import BertTokenizer


def read_data(file):
    data = []
    with open(file, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            data.append(row)
    return data


def split_dataset(train_size=80, valid_size=10, max_line_no=50000):
    input_file = Path('rus.txt')
    train_file = Path('train.txt')
    valid_file = Path('dev.txt')
    test_file = Path('test.txt')

    line_no = 0
    with input_file.open('r', encoding='utf-8') as inp, \
            train_file.open('w', encoding='utf-8') as train, \
            valid_file.open('w', encoding='utf-8') as valid, \
            test_file.open('w', encoding='utf-8') as test:
        while line_no < max_line_no:
            line = inp.readline()
            if not line:
                break
            if line_no % 100 < train_size:
                target = train
            elif line_no % 100 < train_size + valid_size:
                target = valid
            else:
                target = test

            target.write(line)
            line_no += 1


class TextProcessor:
    def __init__(self, wpp: BertTokenizer) -> None:
        self.wpp = wpp
        self.wpp.max_len = 1e12

    def save_to_dir(self, save_dir: Path) -> None:
        (save_dir / META_FILE).write_text(json.dumps({
            'do_lower_case': self.wpp.basic_tokenizer.do_lower_case
        }))
        self.wpp.save_pretrained(str(save_dir.resolve()))

    @staticmethod
    def from_pretrained(name_or_path: str, **kwargs) -> 'TextProcessor':
        wpp = BertTokenizer.from_pretrained(name_or_path, **kwargs)
        return TextProcessor(wpp)

    def save_pretrained(self, save_directory: str):
        self.wpp.save_pretrained(save_directory)

    def decode_ids(self, ids: List[int]) -> str:
        return self.wpp.convert_tokens_to_string(self.wpp.convert_ids_to_tokens(ids, skip_special_tokens=True))

    def encode_as_ids(self, text: str) -> List[int]:
        return self.wpp.convert_tokens_to_ids(self.wpp.tokenize(text))

    def encode_as_pieces(self, text: str) -> List[str]:
        return self.wpp.tokenize(text)

    @property
    def bos_id(self) -> int:
        return int(self.wpp.vocab['[CLS]'])

    @property
    def eos_id(self) -> int:
        return int(self.wpp.vocab['[SEP]'])

    @property
    def pad_id(self) -> int:
        return int(self.wpp.vocab['[PAD]'])

    @property
    def mask_id(self) -> int:
        return int(self.wpp.vocab['[MASK]'])

    @property
    def unk_id(self) -> int:
        return int(self.wpp.vocab['[UNK]'])

    @property
    def vocab_size(self) -> int:
        return len(self.wpp.vocab) + len(self.wpp.additional_special_tokens)


class RawWikiPair(NamedTuple):
    src: str
    tgt: str


class WikiPairsDatasetItem(NamedTuple):
    src_tokens: List[int]
    tgt_tokens: List[int]
    src_words: int
    tgt_words: int
    src_len: int
    tgt_len: int


def read_wiki_pairs(path: Path) -> Generator[RawWikiPair, None, None]:
    with path.open('rt') as f:
        for line in csv.reader(f, delimiter='\t'):
            src, tgt = line[0].strip(), line[1].strip()
            pairs = RawWikiPair(src, tgt)
            yield pairs


def encode_sentence(sentence: str,
                    text_processor: TextProcessor,
                    cutoff: int = 200,
                    extend: bool = True) -> Tuple[List[int], int, int]:
    encoded_words = text_processor.encode_as_ids(sentence)
    split_sentence = text_processor.decode_ids(encoded_words).split(' ')
    if len(encoded_words) > cutoff:
        encoded_words = encoded_words[:cutoff]
        split_sentence = text_processor.decode_ids(encoded_words).split(' ')
    result_len = len(encoded_words)
    if extend and result_len < cutoff:
        encoded_words.extend([0] * (cutoff - result_len))
    num_words = len(split_sentence)
    return encoded_words, result_len, num_words


def extend_upto_cutoff(tokens: List[int], cutoff: int, pad: int = 0) -> List[int]:
    length = len(tokens)
    if length < cutoff:
        tokens.extend([pad] * (cutoff - length))
    return tokens


def encode_wiki_pair(pair: RawWikiPair, text_processor: TextProcessor, cutoff: int = 200) -> WikiPairsDatasetItem:
    src_tokens, src_len, src_words = encode_sentence(pair.src, text_processor, cutoff // 2, extend=False)
    tgt_tokens, tgt_len, tgt_words = encode_sentence(pair.tgt, text_processor, cutoff // 2, extend=False)
    src_tokens = extend_upto_cutoff(src_tokens, cutoff)
    tgt_tokens = extend_upto_cutoff(tgt_tokens, cutoff)
    return WikiPairsDatasetItem(src_tokens, tgt_tokens, src_words, tgt_words, src_len, tgt_len)


class WikiPairsDataset(Dataset):
    def __init__(self, items: List[WikiPairsDatasetItem]):
        super(WikiPairsDataset, self).__init__()
        self.items = items

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            'src_tokens': torch.tensor(item.src_tokens, dtype=torch.long),
            'tgt_tokens': torch.tensor(item.tgt_tokens, dtype=torch.long),
            'src_words': item.src_words,
            'tgt_words': item.tgt_words,
            'src_len': item.src_len,
            'tgt_len': item.tgt_len,
        }

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def from_path(path: Path, text_processor: TextProcessor, *, cutoff: int = 200) -> 'WikiPairsDataset':
        items: List[WikiPairsDatasetItem] = []
        for pair in read_wiki_pairs(path):
            items.append(encode_wiki_pair(pair, text_processor, cutoff))
        return WikiPairsDataset(items)


def insert_bounds(seqs: torch.Tensor, lens: torch.Tensor,
                  start_code: Optional[int] = None,
                  end_code: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    change = 0
    start = 0
    if start_code is not None:
        change += 1
        start = 1
    if end_code is not None:
        change += 1

    new_shape = (seqs.shape[0], seqs.shape[1] + change)
    new_seqs = torch.zeros(new_shape, dtype=seqs.dtype, device=seqs.device)
    new_lens = lens.clone()

    new_seqs[:, start:start + seqs.shape[1]] = seqs
    new_lens[:] += change

    if start_code is not None:
        new_seqs[:, 0] = start_code
    if end_code is not None:
        new_seqs[range(lens.shape[0]), lens + change - 1] = end_code

    return new_seqs, new_lens


class Batch:
    def __init__(self, seqs: torch.Tensor, lens: torch.Tensor) -> None:
        if torch.numel(lens) > 0:
            max_len = cast(int, lens.max().item())
            seqs = seqs[:, :max_len]
        self.seqs = seqs
        self.lens = lens

    def insert_bounds(self, start_code: Optional[int] = None, end_code: Optional[int] = None) -> 'Batch':
        new_seqs, new_lens = insert_bounds(self.seqs, self.lens, start_code=start_code, end_code=end_code)
        return Batch(new_seqs, new_lens)

    def to(self, device: torch.device) -> 'Batch':
        return Batch(self.seqs.to(device), self.lens.to(device))

    def __len__(self):
        return self.lens.shape[0]

    @property
    def device(self):
        return self.seqs.device


def parse_batch(batch: Dict[str, Any], start_code: int, end_code: int) -> Tuple[Batch, Batch]:
    src_seqs = Batch(batch['src_tokens'], batch['src_len'])
    tgt_seqs = Batch(batch['tgt_tokens'], batch['tgt_len'])
    tgt_seqs = tgt_seqs.insert_bounds(start_code=start_code)
    return src_seqs, tgt_seqs


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(LMHead, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer = get_clones(ResidualConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.layer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.layer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.layer = get_clones(ResidualConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.layer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.layer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.layer[2](x, self.feed_forward)


def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_enc = torch.zeros((1, max_len, d_model))
        periods = torch.pow(10000, -2 * torch.arange(d_model // 2 + 1, dtype=torch.float) / d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        pos_enc[0, :, 0::2] = torch.sin(periods[:(d_model + 1) // 2] * positions)
        pos_enc[0, :, 1::2] = torch.cos(periods[:d_model // 2] * positions)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return self.dropout(x)


class TransformerConfig:
    def __init__(self,
                 d_model=20,
                 nhead=5,
                 num_encoder_layers=5,
                 num_decoder_layers=5,
                 dim_feedforward=80,
                 dropout=0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, json_config):
        config = cls()
        for key, value in json_config.items():
            config.__dict__[key] = value
        return config


def new_parameter(data=None, requires_grad=True):
    return torch.nn.Parameter(data=data, requires_grad=requires_grad)


class Transformer(nn.Module):

    def __init__(self, vocab_size: int, config: TransformerConfig, start_code: int):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        self.start_code = start_code
        attn = MultiHeadedAttention(config.nhead, config.d_model)
        ff = PositionwiseFeedForward(config.d_model, config.dim_feedforward, config.dropout)
        emb = nn.Embedding(vocab_size, config.d_model)
        pos_enc = PositionalEncoding(config.d_model, config.dropout)

        encoder = Encoder(EncoderLayer(config.d_model, c(attn), c(ff), config.dropout), config.num_encoder_layers)
        decoder = Decoder(DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
                          config.num_decoder_layers)
        src_embed, tgt_embed = get_clones(nn.Sequential(c(emb), c(pos_enc)), 2)
        generator = LMHead(config.d_model, vocab_size)

        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

        self._reset_parameters()

        self.model.tgt_embed[0].weight = new_parameter(self.model.src_embed[0].weight.clone(), True)
        self.model.generator.linear.weight = new_parameter(self.model.src_embed[0].weight.clone(), True)

        self.d_model = config.d_model
        self.nhead = config.nhead

    @property
    def device(self) -> torch.device:
        return self.model.generator.linear.weight.device

    def forward(self, src, tgt):
        tgt_mask = generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        out = self.model(src, tgt, None, tgt_mask)
        return out

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def start_generation(self, src: Tensor):
        initial = GenerationState(self, src)
        return initial.next(self.start_code)


@jit.script
def create_mask(target: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(target.size(1), dtype=lens.dtype, device=lens.device).repeat(target.size(0), 1) < lens.unsqueeze(1)
    for i in range(len(target.size()) - 2):
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=torch.bool)


def cross_entropy(out: Tensor, expected: Tensor) -> Tensor:
    return F.cross_entropy(
        out.view(-1, out.shape[-1]),
        expected.view(-1),
        reduction='none').view(expected.shape)


def cross_entropy_label_smoothing(out: Tensor, expected: Tensor, eps: float = 0.) -> Tensor:
    if eps == 0.:
        result = cross_entropy(out, expected)
    else:
        n_classes = out.size(-1)
        delta = torch.zeros_like(out.view(-1, out.shape[-1])).scatter(1, expected.view(-1, 1), 1)
        q = delta * (1 - eps) + (torch.tensor(1.) - delta) * eps / (n_classes - 1)
        log_p = F.log_softmax(out, -1)
        result = - (q * log_p.view(-1, out.shape[-1])).sum(dim=1)
        result = result.view(expected.shape)
    return result


def batch_cross_entropy_tensor(out: torch.Tensor, expected: torch.Tensor, exp_lens: torch.Tensor, mask_delta=0,
                               eps: float = 0.) -> torch.Tensor:
    loss = cross_entropy_label_smoothing(out, expected, eps)
    mask = create_mask(loss, exp_lens + torch.tensor(mask_delta, device=exp_lens.device))
    mask.requires_grad = False
    loss.masked_fill_(~mask, 0.0)
    return loss


def batch_cross_entropy(out: Batch, expected: Batch, eps: float = 0.) -> torch.Tensor:
    return batch_cross_entropy_tensor(out.seqs, expected.seqs, expected.lens, eps=eps).sum() / out.lens.sum()


def choose_device():
    if torch.cuda.device_count() > 0:
        device_no = torch.cuda.current_device()
        return torch.device(f'cuda:{device_no}')
    else:
        return torch.device('cpu')


def random_subset(dataset: Dataset, samples: int) -> Dataset:
    return Subset(
        dataset,
        torch.randperm(n=len(dataset))[:samples].tolist())


def setup_datasets(text_processor: TextProcessor) -> Tuple[WikiPairsDataset, List[Tuple[str, Dataset]]]:
    train_dataset = WikiPairsDataset.from_path(Path('train.txt'), text_processor, cutoff=200)
    dev_dataset = WikiPairsDataset.from_path(Path('dev.txt'), text_processor, cutoff=200)
    train_eval_dataset = random_subset(train_dataset, len(dev_dataset))
    return train_dataset, [('train', train_eval_dataset), ('dev', dev_dataset)]


def setup_model(vocab_size, config, device: torch.device, start_code) -> nn.Module:
    return Transformer(vocab_size, config, start_code).to(device)


def setup_optimizer(module: nn.Module) -> Tuple[Optimizer, ExponentialLR]:
    optimizer = Adam(module.parameters(), lr=1e-3)
    return optimizer, ExponentialLR(optimizer, gamma=0.99)


def setup_text_processor() -> TextProcessor:
    return TextProcessor.from_pretrained(Path('bert-base-uncased-vocab.txt'))


def create_loader(data: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True)


def normalize_step(batch_num: int, batches: int, epoch: int, base: int = 1000):
    return batch_num * base // batches + base * epoch


def device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            res[name] = value.to(device)
        else:
            res[name] = value
    return res


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(log_dir: Path, epoch: int, model: nn.Module) -> None:
    model_path = log_dir / f'model-{epoch}'
    print(f'Saving model to {str(model_path)} ...')

    with model_path.open('bw') as f:
        torch.save({
            'module': model.state_dict()
        }, f)
    print('Done')


class GenerationState:

    def __init__(self, module: Transformer, src: Tensor, memory: Optional[Tensor] = None,
                 tgt: Optional[Tensor] = None) -> None:
        self.module = module
        self.src = src
        if memory is None:
            self.memory = self.module.model.encode(src, None)
        else:
            self.memory = memory
        self.tgt = tgt

    @torch.no_grad()
    def next(self, token: int):
        tgt = torch.tensor([[token]], device=self.module.device)
        if self.tgt is not None:
            self.tgt = torch.cat([self.tgt, tgt], dim=1)
        else:
            self.tgt = tgt
        tgt_mask = generate_square_subsequent_mask(self.tgt.shape[1]).to(self.tgt.device)
        logits = self.module.model.decode(tgt, self.memory, None, tgt_mask)
        scores = self.module.model.generator(logits)
        return GenerationState(self.module, self.src, self.memory, self.tgt), scores


def presample_prefix(start: Tuple[GenerationState, torch.Tensor], prefix: Optional[List[int]]) -> Tuple[GenerationState, torch.Tensor]:
    state, scores = start
    if prefix:
        for token in prefix:
            state, scores = state.next(token)
    return state, scores


def complete_greedily(start: Tuple[GenerationState, torch.Tensor], max_len: int,
                      prefix: Optional[List[int]] = None, take_max: bool = False,
                      end_code: int = 0) -> Tuple[List[int], GenerationState]:
    state, scores = presample_prefix(start, prefix)

    result: List[int] = []
    while len(result) < max_len:
        if take_max:
            token = int(torch.argmax(scores).item())
        else:
            probs = F.softmax(scores, dim=-1)
            token = int(dist.Categorical(probs=probs).sample().item())
        if token == end_code:
            break
        result.append(token)
        state, scores = state.next(token)
    return result, state


def translate_text(model, text_processor, prompt: str, text_len: int = 128) -> str:
    prefix = text_processor.encode_as_ids(prompt)
    prefix_tensor = torch.tensor(prefix, device=model.device).unsqueeze(0)
    start = model.start_generation(prefix_tensor)
    generated = complete_greedily(start=start, max_len=text_len, prefix=prefix, end_code=text_processor.eos_id)[0]
    text = text_processor.decode_ids(generated)
    return text


def train_model(epochs=50, batch_size=50):
    device = choose_device()
    fix_seed(42)

    text_processor = setup_text_processor()
    bos_id, eos_id = text_processor.bos_id, text_processor.eos_id
    train_dataset, evals = setup_datasets(text_processor)

    config = TransformerConfig()
    model = setup_model(text_processor.vocab_size, config, device, text_processor.bos_id)
    optimizer, scheduler = setup_optimizer(model)
    save_model(Path('model'), 0, model)

    for epoch in range(epochs):

        train_loader = create_loader(train_dataset, batch_size=batch_size)
        total_batches = len(train_loader)

        model.train()
        epoch_loss = 0
        for batch_no, batch in enumerate(tqdm(train_loader, ncols=40, desc=f'Epoch {epoch}')):
            optimizer.zero_grad()
            batch = device_batch(batch, device)
            src, tgt = parse_batch(batch, bos_id, eos_id)
            scores = model(src.seqs, tgt.seqs)
            seqs = model.model.generator(scores)
            loss = batch_cross_entropy(Batch(seqs, tgt.lens), tgt, eps=0.1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= total_batches
        print(f'loss: {epoch_loss}')
        prompt = 'Hello world'
        translation = translate_text(model, text_processor, prompt)
        print(f'prompt: {prompt} translation: {translation}')
        save_model(Path('model'), epoch, model)


if __name__ == '__main__':
    train_model()
