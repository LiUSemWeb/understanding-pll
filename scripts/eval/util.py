import json
from itertools import permutations
from collections import Counter, defaultdict

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


def mentions_for(doc, entity):
    return set(m['name'] for m in doc['vertexSet'][entity])


def entities(doc):
    ents = {}
    for i, ent in enumerate(doc['vertexSet']):
        ents[i] = ent[0]['name']
    return ents


def prompt(rel_info, rel, xy=True):
    if xy:
        return rel_info[rel]['prompt_xy']
    else:
        return rel_info[rel]['prompt_yx']


def candidate_maps(prompt:str, doc):
    choices, types = mentions(doc)
    prompts = {}
    for a, b in permutations(choices, 2):
        prompts[prompt.replace("?x", a, 1).replace("?y", b, 1)] = ((a, types[a]), (b, types[b]))
    return prompts


def candidates(prompt:str, doc):
    choices = mentions(doc)[0]
    prompts = []
    if prompt[-1] != ".":
        prompt += "."
    for a, b in permutations(choices, 2):
        prompts.append(prompt.replace("?x", a, 1).replace("?y", b, 1))
    return prompts


def answers(doc):
    ents = entities(doc)
    if 'labels' in doc:
        ans = []
        for an in doc['labels']:
            ans.append((ents[an['h']], an['r'], ents[an['t']]))
        return ans


def possible_answers(doc):
    ans = []
    out = []
    if 'labels' in doc:
        for an in doc['labels']:
            ans.append((mentions_for(doc, an['h']), an['r'], mentions_for(doc, an['t'])))
    for hs, r, ts in ans:
        for h in hs:
            for t in ts:
                out.append((h, r, t))
    return out


def read_docred(dset:str='dev', *, path:str='data/docred', doc=-1, limit=0, verbose=False):
    # print(os.getcwd())
    if dset == 'train':
        dset = 'train_annotated'
    with open(f"{path}/{dset}.json", encoding='utf8') as datafile:
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
                print("\n"+"\n".join(candidates("The headquarters of ***mask*** is in ***mask***.", doc)))
                print("-" * 50)
                print("Ans:", "\n"+"\n".join(str(a) for a in answers(doc)))
                print("-" * 50)
            yield doc
        else:
            if limit > 0:
                for i, doc in enumerate(jfile):
                    if i < limit:
                        yield doc
            else:
                yield from jfile


def get_rel_info(data_path="/data/git/text2kg2023-rilca-util/res/eswc2023-results"):
    with open(f'{data_path}/rel_info_full.json', 'r', encoding='utf8') as rel_info_file:
        rel_info = json.load(rel_info_file)
    return rel_info


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


def domain_range(working_dir="/data/git/text2kg2023-rilca-util", docred=None, rel_info=None, thresh=0.05):
    # dir = "/data/git/text2kg2023-rilca-util"#\\res\\eswc2023-results"
    if not docred:
        docred = read_docred(dset='train', path=f"{working_dir}/data")
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
    print(domain_range())