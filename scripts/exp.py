import json
import os.path
import time
from itertools import permutations
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functional import pseq, seq
import torch
from torch import nn
from tqdm import tqdm
from collections import Counter, defaultdict
import sys

# from eval.util import domain_range

torch.set_num_threads((torch.get_num_threads()*2) - 1)
torch.set_num_interop_threads((torch.get_num_interop_threads()*2) - 1)

BATCH_SIZE = 512
# If you have memory issues with some documents, you can add them here to skip. 332 is particularly difficult.
TOO_BIG = []#[332]
SENTS_PER_BATCH = 32


class ScoringMethod(nn.Module):
    def __init__(self, label):
        super(ScoringMethod, self).__init__()
        self.label = label


class PllScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(PllScoringMethod, self).__init__(label)

    def forward(self, probs, origids=None, return_all=False, **kwargs):
        slen = len(probs) - 1
        dia = torch.diag(probs[1:].gather(-1, origids.unsqueeze(0).repeat(slen, 1).unsqueeze(-1)).squeeze(-1), diagonal=0)
        dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob


class ComparativeScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(ComparativeScoringMethod, self).__init__(label)

    def forward(self, probs, return_all=False, **kwargs):
        slen = len(probs) - 1
        dia = self.calc(probs[0, :slen], probs[torch.arange(1, slen + 1), torch.arange(slen)])
        dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob

    def calc(self, p: torch.tensor, q: torch.tensor):
        raise NotImplementedError


class JSD(ComparativeScoringMethod):
    def __init__(self):
        super(JSD, self).__init__("jsd")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def calc(self, p: torch.tensor, q: torch.tensor):
        m = torch.log_((0.5 * (p + q)))
        return 1 - (0.5 * (torch.sum(self.kl(m, p.log()), dim=-1) + torch.sum(self.kl(m, q.log()), dim=-1)))


class PLL(PllScoringMethod):
    def __init__(self):
        super(PLL, self).__init__("pll")


class CSD(ComparativeScoringMethod):
    def __init__(self):
        super(CSD, self).__init__("csd")
        self.csd = torch.nn.CosineSimilarity(dim=1)

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.csd(p, q)


class ESD(ComparativeScoringMethod):
    def __init__(self):
        super(ESD, self).__init__("esd")
        self.pwd = torch.nn.PairwiseDistance()
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def norm(self, dist):
        return (torch.relu(self.sqrt - dist) + 0.000001) / self.sqrt

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.norm(self.pwd(p, q))


class MSD(ComparativeScoringMethod):
    def __init__(self):
        super(MSD, self).__init__("msd")
        self.mse = torch.nn.MSELoss(reduction="none")

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.mse(p, q).mean(axis=-1)


class HSD(ComparativeScoringMethod):
    def __init__(self):
        super(HSD, self).__init__("hsd")
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def calc(self, p: torch.tensor, q: torch.tensor):
        return 1 - torch.sqrt_(torch.sum(torch.pow(torch.sqrt_(p) - torch.sqrt_(q), 2), dim=-1)) / self.sqrt


KNOWN_METHODS = [CSD(), ESD(), JSD(), MSD(), HSD(), PLL()]
KNOWN_METHODS = {m.label: m for m in KNOWN_METHODS}


def sentences(doc):
    for sent in doc['sents']:
        yield " ".join(sent)


def mentions(doc):
    ments = set()
    types = dict()
    for ent in doc['vertexSet']:
        # print(ent)
        for ment in ent:
            name = ment['name']
            ments.add(name)
            if name not in types:
                types[name] = ment['type']
    return list(ments), types


def entities(doc):
    ents = {}
    for i, ent in enumerate(doc['vertexSet']):
        ents[i] = ent[0]['name']
    return ents


def prompt(rel, xy=True, ensure_period=True):
    if xy:
        prompt = rel_info[rel]['prompt_xy']
    else:
        prompt = rel_info[rel]['prompt_yx']
    if ensure_period and prompt[-1] != '.':
        return prompt + "."
    else:
        return prompt


def candidates(prompt:str, choices, return_ments=False):
    for a, b in permutations(choices, 2):
        if return_ments:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1), (a, b)
        else:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1)


def candidate_maps(prompt:str, doc):
    choices, types = mentions(doc)
    prompts = {}
    for a, b in permutations(choices, 2):
        prompts[prompt.replace("?x", a, 1).replace("?y", b, 1)] = ((a, types[a]), (b, types[b]))
    return prompts


def answers(doc):
    ents = entities(doc)
    if 'labels' in doc:
        ans = []
        for an in doc['labels']:
            ans.append((ents[an['h']], an['r'], ents[an['t']]))
        return ans


def answer_prompts(doc):
    ents = entities(doc)
    if 'labels' in doc:
        ans = []
        for an in doc['labels']:
            ans.append(prompt(an['r']).replace("?x", ents[an['h']], 1).replace("?y", ents[an['t']], 1))
        return ans


def read_docred(dset:str='dev', *, path:str='data/docred', doc=-1, verbose=False):
    if dset == 'train':
        dset = 'train_annotated'
    with open(f"{path}/{dset}.json") as datafile:
        jfile = json.load(datafile)

        if doc >= 0:
            doc = jfile[doc]
            if verbose:
                print("Title:", doc['title'])
                print("-" * 50)
                if dset != 'test':
                    print("Labels:", doc['labels'])
                    print("-" * 50)
                print("VSet:")
                for v in doc['vertexSet']:
                    print("   ", v)
                print("-" * 50)
                print("Entities:", entities(doc))
                print("-" * 50)
                print("Sents:", "\n"+"\n".join(sentences(doc)))
                print("-" * 50)
                print("\n"+"\n".join(candidates("The headquarters of ***mask*** is in ***mask***.", mentions(doc)[0])))
                print("-" * 50)
                print("Ans:", "\n"+"\n".join(str(a) for a in answers(doc)))
                print("-" * 50)
            yield doc
        else:
            yield from jfile


# scores equivalently to the old method, even with padding.
# Can be used to batch across examples.
def pll_score_batched(self, sents: list, return_all=False):
    self.bert.eval()
    key_to_sent = {}
    with torch.no_grad():
        data = {}
        for sent in sents:
            tkns = self.tokenizer.tokenize(sent)
            data[len(data)] = {
                'tokens': tkns,
                'len': len(tkns)
            }
        scores = {"pll": {}}
        all_plls = {"pll": {}}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        methods = [PLL()]

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        # del all_probs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores)["pll"], unsort_flatten(all_plls)["pll"]
        return unsort_flatten(scores)["pll"]


def _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all):
    bert_forward = torch.concat(
        [torch.tensor(self.mask_tokenize(data[d]['tokens'],
                                         keep_original=True,
                                         pad=max(lens) - data[d]['len']
                                         ),
                      device="cpu"
                      )
         for d in inds],
        dim=0).to(self.device)
    _probs = self.softmax(self.bert_am(bert_forward)[0])[:, 1:, :]
    del bert_forward

    use_pll = "pll" in [method.label for method in methods]

    for ind, slen in zip(inds, lens):
        origids = torch.tensor(self.tokenizer.convert_tokens_to_ids(data[ind]['tokens']), dtype=torch.long).to(
            self.device) if use_pll else None

        for method in methods:
            prob, alls = method(_probs[:slen + 1], origids=origids, return_all=True)
            if return_all:
                assert ind not in all_plls[method.label]
                all_plls[method.label][ind] = alls
            assert ind not in scores[method.label]
            scores[method.label][ind] = prob
            del alls, prob
        _probs = _probs[slen + 1:]
    del _probs


def unsort_flatten(mapping):
    # print(mapping.keys())
    return {f: list(mapping[f][k] for k in range(len(mapping[f]))) for f in mapping}


def cos_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[CSD()], sents=sents, return_all=return_all)


def euc_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[ESD()], sents=sents, return_all=return_all)


def jsd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[JSD()], sents=sents, return_all=return_all)


def msd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[MSD()], sents=sents, return_all=return_all)


def hel_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[HSD()], sents=sents, return_all=return_all)


def all_score_batched(self, sents: list,  return_all=True):
    return score_batched(self, methods=list(KNOWN_METHODS.values()), sents=sents, return_all=return_all)


def score_batched(self, methods, sents: list, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        data = {}
        for sent in sents:
            # Tokenize every sentence
            tkns = self.tokenizer.tokenize(sent)
            data[len(data)] = {
                'tokens': tkns,
                'len': len(tkns)
            }

        scores = {m.label: {} for m in methods}
        all_plls = {m.label: {} for m in methods}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores), unsort_flatten(all_plls)
        return unsort_flatten(scores)


def print_token_details(*, dset, data_path, resdir="res", model="bert-large-cased", fb=None):
    if fb is None:
        fb = FitBert(disable_gpu=False, model_name=model)
        fb.tokenizer.add_tokens(['?x', '?y'])
    lockfile = os.path.join(resdir, ".lock")

    for dset in [dset]:  # ['dev', 'train']:
        tqdocs = tqdm(read_docred(dset, path=data_path), total=(1000 if dset == 'dev' else 3053))
        for d, doc in enumerate(tqdocs):
            while os.path.exists(lockfile):
                print("Sleep.")
                time.sleep(1.0)

            # Make lock file
            with open(lockfile, 'w'):
                # for every label:
                # look for a corresponding file.
                mtag = "" if model == "bert-large-uncased" else f"{model}_"
                fname = f'{resdir}/{mtag}{dset}_{d}_tokens.csv'
                # if not exists: make file, add to array.
                if os.path.exists(fname):
                    print(f"Skipping doc {d}")
                    os.remove(lockfile)
                    continue
            try:
                os.remove(lockfile)
                ments = mentions(doc)[0]
                answs = answer_prompts(doc)
                trels = len(rel_info)
                with open(fname, 'w') as csvfile:
                    for r, p in enumerate(rel_info):
                        prom = prompt(p)
                        mapping = candidate_maps(prom, doc)
                        csvfile.write(f"{dset} {d} {p}\n")
                        cands = list(candidates(prom, ments, return_ments=True))
                        tcands = len(ments)
                        tcands *= (tcands - 1)
                        scores = []
                        alloweds = []
                        plls = []
                        verbs = []
                        tqdocs.set_description(f"D{d} {p}[{r}/{trels}] C[{tcands}]")
                        _prompt = fb.tokenizer.tokenize(prom)
                        for can, (head, tail) in cands:
                            scores.append(can in answs)  # Are they true sentences?
                            x, y = mapping[can]
                            alloweds.append((x[1] in rel_info[p]['domain']) and (y[1] in rel_info[p]['range']))
                            _tokenz = fb.tokenizer.tokenize(can)
                            _head = fb.tokenizer.tokenize(head)
                            _tail = fb.tokenizer.tokenize(tail)
                            x_loc = _prompt.index('?x')
                            y_loc = _prompt.index('?y') + len(_head) - 1
                            verb_loc = rel_info[p]['verb']
                            verb_bump = 0
                            if verb_loc > x_loc:
                                verb_bump += len(_head) - 1
                            if verb_loc > y_loc:
                                verb_bump += len(_tail) - 1
                            verbs.append(verb_loc + verb_bump)

                            bios = []
                            for t in range(len(_tokenz)):
                                if t in range(x_loc, x_loc + len(_head)):
                                    if t == x_loc:
                                        bios.append('B')
                                    else:
                                        bios.append('I')
                                elif t in range(y_loc, y_loc + len(_tail)):
                                    if t == y_loc:
                                        bios.append('B')
                                    else:
                                        bios.append('I')
                                else:
                                    bios.append('O')

                            plls.append(bios)  # B/I/O markers
                        for c, v, s, a, pl in sorted(zip(cands, verbs, scores, alloweds, plls), key=lambda x: x[2]):
                            plls_string = ", ".join(pl)
                            csvfile.write(f"\"{c[0]}\", {a}, {v}, {s}, {plls_string}\n")
                        csvfile.write(f"\n")
                        csvfile.flush()
            except BaseException as e:
                # If anything went wrong, or CTRL+C was pressed, remove the result files.
                try:
                    fname = f'{resdir}/{dset}_{d}_tokens.csv'
                    os.remove(fname)
                except:
                    pass
                raise e
    return fb

# Borrowed the starter code from https://github.com/writerai/fitbert
# Heavily modified, assume there are some strange changes ahead
class FitBert:
    def __init__(
            self,
            model_name="bert-large-uncased",
            disable_gpu=False,
    ):
        # self.mask_token = mask_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        # self._score = pll_score_batched
        print("device:", self.device)

        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.bert.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token = self.tokenizer.mask_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

    def __call__(self, *args, **kwds):
        return self.bert(*args, **kwds)

    def bert_am(self, data, *args, **kwds):
        return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

    def tokenize(self, *args, **kwds):
        return self.tokenizer(*args, **kwds)

    def mask_tokenize(self, tokens, keep_original=False, pad=0):
        # tokens = self.tokenize(sent)
        if keep_original:
            return [self._tokens(tokens, pad=pad)] + self.mask_tokenize(tokens, keep_original=False, pad=pad)
        else:
            return (seq(tokens)
                    .enumerate()
                    .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i, pad=pad))
                    .list()
                    )

    def _tokens_to_masked_ids(self, tokens, mask_ind, pad=0):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = self.mask_token
        masked_ids = self._tokens(masked_tokens, pad=pad)
        return masked_ids

    def _tokens(self, tokens, pad=0):
        tokens = [self.cls_token] + tokens + [self.sep_token] + [self.pad_token] * pad
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @staticmethod
    def softmax(x):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        # Further reduce memory impact by making it an in-place operation. Beware.
        return FitBert._inn_soft(x.exp_())

    @staticmethod
    def _inn_soft(xexp):
        return xexp / (xexp.sum(-1)).unsqueeze(-1)


def fitb(dset, d_start, d_end, data_path, metric="pll", resdir="res"):
    metrics = {
        "pll": pll_score_batched,
        "cos": cos_score_batched,
        "euc": euc_score_batched,
        "jsd": jsd_score_batched,
        "msd": msd_score_batched,
        "hel": hel_score_batched,
        "all": all_score_batched,
    }

    if metric not in metrics:
        print(f"Metric {metric} not found. Options are {', '.join(metrics.keys())}.")
        return

    score_func = metrics[metric]

    fb = FitBert(disable_gpu=False)
    data_path = data_path if data_path else "data/docred"

    for dset in [dset]:# ['dev', 'train']:
        tqdocs = tqdm(read_docred(dset, path=data_path), total=(1000 if dset == 'dev' else 3053))
        for d, doc in enumerate(tqdocs):
            if d not in range(d_start, d_end):
                continue
            ments = mentions(doc)[0]
            trels = len(rel_info)
            with open(f'{resdir}/{dset}_{d}_{metric}s.csv', 'w') as csvfile:
                for r, p in enumerate(rel_info):
                    if r > 10:
                        continue
                    csvfile.write(f"{dset} {d} {p}\n")
                    cands = list(candidates(prompt(p), ments))
                    tcands = len(ments)
                    tcands *= (tcands - 1)
                    scores = []
                    plls = []
                    tqdocs.set_description(f"D{d} {p}[{r}/{trels}] C[{tcands}]")
                    i = 0
                    while i < len(cands):
                        scs, pls = score_func(fb, cands[i:i + SENTS_PER_BATCH], return_all=True)
                        scores.extend(scs)
                        plls.extend(pls)
                        i += SENTS_PER_BATCH
                    for c, s, pl in sorted(zip(cands, scores, plls), key=lambda x: x[1]):
                        plls_string = ", ".join(f"{v:.6f}" for v in pl)
                        csvfile.write(f"\"{c}\", {s:.2f}, {plls_string}\n")
                    csvfile.write(f"\n")
                    csvfile.flush()


def fitb_distributed(dset, data_path, resdir="res", model="bert-large-uncased", fb=None):
    score_func = score_batched
    trels = len(rel_info)

    if fb is None:
        fb = FitBert(disable_gpu=False, model_name=model)
    data_path = data_path if data_path else "data/docred"
    mtag = "" if model == "bert-large-uncased" else f"{model}_"

    lockfile = os.path.join(resdir, ".lock")

    locked = False
    lock = None

    for dset in [dset]:  # ['dev', 'train']:
        # while lock file exists:
        #   sleep 1 second
        tqdocs = tqdm(read_docred(dset, path=data_path), total=(1000 if dset == 'dev' else 3053))
        for d, doc in enumerate(tqdocs):
            if d in TOO_BIG:
                continue
            torch.cuda.empty_cache()
            ments = mentions(doc)[0]

            file_handles = {}
            metrics = []
            if not locked:
                while os.path.exists(lockfile):
                    print("Sleep.")
                    time.sleep(1.0)
                lock = open(lockfile, 'w')
                locked = True

            # for every label:
            for metric in KNOWN_METHODS:
                # look for a corresponding file.
                fname = f'{resdir}/{mtag}{dset}_{d}_{metric}s.csv'
                # if not exists: make file, add to array.
                if not os.path.exists(fname):
                    file_handles[metric] = open(fname, 'w')
                    metrics.append(metric)
            # if array empty: skip to next number
            if len(metrics) == 0:
                print(f"Skipping doc {d}")
                continue
            try:
                lock.close()
                os.remove(lockfile)
                locked = False
                lock = None
                methods = [KNOWN_METHODS[m] for m in metrics]
                # print(f"Running for {metrics}")
                for r, p in enumerate(rel_info):
                    for fh in file_handles.values():
                        fh.write(f"{dset} {d} {p}\n")
                    cands = list(candidates(prompt(p), ments))
                    tcands = len(ments)
                    tcands *= (tcands - 1)
                    scores = {m:[] for m in metrics}
                    plls = {m:[] for m in metrics}
                    tqdocs.set_description(f"D{d} {p}[{r}/{trels}] C[{tcands}]")
                    i = 0
                    while i < len(cands):
                        if len(methods) > 0:
                            scs, pls = score_func(fb, methods=methods, sents=cands[i:i + SENTS_PER_BATCH], return_all=True)
                            torch.cuda.empty_cache()
                            for metric in metrics:
                                scores[metric].extend(scs[metric])
                                plls[metric].extend(pls[metric])
                        i += SENTS_PER_BATCH
                    for metric in metrics:
                        for c, s, pl in sorted(zip(cands, scores[metric], plls[metric]), key=lambda x: x[1]):
                            plls_string = ", ".join(f"{v:.6f}" for v in pl)
                            file_handles[metric].write(f"\"{c}\", {s:.2f}, {plls_string}\n")
                        file_handles[metric].write(f"\n")
                        file_handles[metric].flush()
            except BaseException as e:
                # If anything went wrong, or CTRL+C was pressed, remove the result files.
                for metric in metrics:
                    try:
                        fname = f'{resdir}/{mtag}{dset}_{d}_{metric}s.csv'
                        os.remove(fname)
                    except:
                        pass
                raise e
            finally:
                for fh in file_handles.values():
                    try:
                        fh.close()
                    except:
                        pass
        if lock is not None:
            lock.close()
        if locked:
            try:
                os.remove(lockfile)
            except:
                pass
    return fb


def get_entity_type(doc, index):
    types = set()
    # print(doc)
    # print(index)
    for mention in doc["vertexSet"][index]:
        types.add(mention["type"])
    return types


def threshold_counter(ct: Counter, thresh=0.05):
    total = sum(ct.values())
    return [a for a, b in ct.items() if b/total > thresh]


def domain_range(docred=None, rel_info=None, thresh=0.05):
    # dir = "/data/git/text2kg2023-rilca-util"#\\res\\eswc2023-results"
    if not docred:
        docred = read_docred(dset='train')
    domain = defaultdict(Counter)
    range = defaultdict(Counter)
    for d, doc in enumerate(docred):
        for label in doc["labels"]:
            r = label['r']
            domain[r].update(get_entity_type(doc, label['h']))
            range[r].update(get_entity_type(doc, label['t']))
    if rel_info:
        for p in domain:
            rel_info[p]['domain'] = threshold_counter(domain[p], thresh)
            rel_info[p]['range'] = threshold_counter(range[p], thresh)
        return rel_info
    else:
        out_domain = {}
        out_range = {}
        for p in domain:
            out_domain[p] = threshold_counter(domain[p], thresh)
            out_range[p] = threshold_counter(range[p], thresh)
        return out_domain, out_range


if __name__ == '__main__':
    dset = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data"
    res_path = sys.argv[3] if len(sys.argv) > 3 else "res"

    BATCH_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    SENTS_PER_BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256

    print(f"Batch size {BATCH_SIZE} of {SENTS_PER_BATCH} each.")
    time.sleep(2.0)

    with open(f'{data_path}/rel_info_full.json', 'r') as rel_info_file:
        rel_info = domain_range(docred=read_docred(dset, path=data_path), rel_info=json.load(rel_info_file))

    torch.cuda.empty_cache()
    for model in ["roberta-base", "bert-base-cased", "roberta-large", "bert-large-cased", "bert-large-uncased"]:
        # First gather all the easy stuff: Information about each sentence, its entities, etc.
        fb = print_token_details(dset=dset, data_path=data_path, resdir=res_path, model=model)
        # Second, run the harder experiments.
        # You should prefer the "distributed" version, as it's generally more robustly implemented.
        fb = fitb_distributed(dset, data_path, resdir=res_path, model=model, fb=fb)
        del fb
        torch.cuda.empty_cache()
        # Sleep a little, let the system unwind a bit. Everyone needs a vacation now and then, right?
        time.sleep(10)

