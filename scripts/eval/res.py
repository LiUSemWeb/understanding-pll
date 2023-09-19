from util import read_docred, candidate_maps, get_rel_info, possible_answers, domain_range
from tqdm import tqdm
from multiprocessing import Pool
import json
import pickle


class DocResult:
    def __init__(self, dset, doc, p, x, y, tx, ty, score, correct=False, allowed=True):
        self.dset = dset
        self.doc = doc
        self.score = float(score)
        self.x = x
        self.y = y
        self.p = p
        self.tx = tx
        self.ty = ty
        self.correct = correct
        self.allowed = allowed

    def __repr__(self):
        return f"[{self.dset}_{self.doc}] {self.score}{'X' if not self.allowed else ''}:{'*' if self.correct else ''} <{self.x}[{self.tx}];{self.p};{self.y}[{self.ty}]>"


def top_k(results, k=1, restrict=False):
    scores = {}
    for dset in results:
        scores[dset] = {}
        for doc in results[dset]:
            for rel in results[dset][doc]:
                if rel not in scores[dset]:
                    scores[dset][rel] = [0, 0]  # hits vs misses
                rset = results[dset][doc][rel]
                if restrict:
                    rset = [r for r in rset if r.allowed]
                if any(r.correct for r in rset):  # Only measure if there can be a right answer
                    top_hit = any(r.correct for r in rset[0:k])
                    scores[dset][rel][0 if top_hit else 1] += 1

    return scores


def prf_thresh(results, threshold):
    tp = 0
    fp = 0
    fn = 0

    for dr in results:
        above = dr.score >= threshold
        correct = dr.correct
        tp += int(above and correct)
        fp += int(above and (not correct))
        fn += int((not above) and correct)
    precision = (tp / (tp + fp)) if tp > 0 else 0
    recall = (tp / (tp + fn)) if tp > 0 else 0
    pr = precision * recall
    f1score = pr/(precision + recall) if pr > 0 else 0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1score
    }
    # return precision, recall, f1score


def prf_all(results, restrict=False):
    # results = flatten(res_out['dev']['P6']) == list of DocResult
    prfs = dict()
    if restrict:
        results = [r for r in results if r.allowed]
    for dr in results:
        thr = dr.score
        if thr not in prfs:
            prfs[thr] = prf_thresh(results, thr)
    return prfs


def flatten(results):
    scores = dict()
    for dset in results:
        scores[dset] = dict()
        for doc in results[dset]:
            for rel in results[dset][doc]:
                if rel not in scores[dset]:
                    scores[dset][rel] = list()  # hits vs misses
                scores[dset][rel].extend(results[dset][doc][rel])
        for rel in scores[dset]:
            scores[dset][rel] = list(sorted(scores[dset][rel], key=lambda r: r.score, reverse=True))

    return scores


def doc_to_entity_mapping(rel_info, doc, prop, direction='xy'):
    return candidate_maps(rel_info[prop][f'prompt_{direction}'], doc)


def read_next_property(dset, fp, res, rel_info, doc, answers):
    prop = None
    header = fp.readline().strip()
    if header:
        prop = header.split()[2]
        if prop == "P1412":  # The prompt was written wrong here...
            line = fp.readline().strip()
            while len(line) > 0:
                line = fp.readline().strip()
            return prop
        else:
            res[prop] = []
        mapping = doc_to_entity_mapping(rel_info, doc, prop)
        # print(f"{len(prop)}|{prop}|{bool(prop)}")
        lines = []
        line = fp.readline().strip()
        while len(line) > 0:
            # print(f"{len(line)}|{line}")
            lines.append(line)
            score, prompt = line.split(":   ")
            if prompt[-1] != '"':  # If we have a new line
                try:
                    while fp.readline().strip()[-1] != '"':
                        continue
                    line = fp.readline().strip()
                    continue  # Skip it.
                except Exception as e:
                    print("\n".join(lines))
                    raise e
            prompt = prompt[1:-1]  # Silly me, leaving in the quotes
            x, y = mapping[prompt]
            # print(score, )
            # dset, doc, p, x, y, tx, ty, score
            correct = (x[0], prop, y[0]) in answers
            allowed = (x[1] in rel_info[prop]['domain']) and (y[1] in rel_info[prop]['range'])
            dr = DocResult(dset, doc['i'], prop, x[0], y[0], x[1], y[1], score, correct, allowed)
            res[prop].append(dr)
            # res[prop].append()
            # print(dr)
            line = fp.readline().strip()
        res[prop] = list(sorted(res[prop], key=lambda r: r.score, reverse=True))
        # res[prop] = lines
    return prop


def repro_topk():
    working_dir = "/data/git/text2kg2023-rilca-util"  # /res/eswc2023-results"
    res_out = {}
    rel_info = domain_range(get_rel_info(data_path=f"{working_dir}/data"))
    # tq = tqdm(total=4053)
    tq = tqdm(total=1000)
    for dset, num in zip(['dev', 'train'], [1000, 3053]):
        if dset == 'train':
            exit(0)
        res_out[dset] = {}
        docred = read_docred(dset=dset, path=f"{working_dir}/data")
        for d, doc in enumerate(docred):
            res_out[dset][d] = {}
            tq.update(1)
            tq.set_description(f"{dset}/{d}")
            doc['i'] = d
            ans = possible_answers(doc)
            parti = f"_P{participant}" if participant > 0 else ""
            filename = f"{working_dir}/res/eswc2023-results/{dset}_{d}{parti}_results.txt"
            # filename = f"{working_dir}/res/eswc2023-results/{dset}_{d}_results.txt"
            with open(filename, 'r', encoding='utf8') as docfile:
                # header = docfile.readline()
                prop = read_next_property(dset, docfile, res_out[dset][d], rel_info, doc, ans)
                while prop:
                    prop = read_next_property(dset, docfile, res_out[dset][d], rel_info, doc, ans)

            if ((d+1)%1000) == 0:
                print("DOC", d)

                for k in range(1, 6):
                    for restrict in [False, True]:

                        results = top_k(res_out, k=k, restrict=restrict)
                        hs = 0
                        ms = 0
                        for p, s in results['dev'].items():
                            if sum(s) > 0:
                                print(f"{p}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")
                            hs += s[0]
                            ms += s[1]
                        s = [hs, ms]
                        print(f"total for {k}{'+dr' if restrict else ''}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")
    tq.close()


def _worker(args):
    d, doc = args
    dset = "dev"
    res_out = {dset: {d: dict()}}
    doc['i'] = d
    ans = possible_answers(doc)
    working_dir = "/data/git/text2kg2023-rilca-util"
    parti = f"_P{participant}" if participant > 0 else ""
    filename = f"{working_dir}/res/eswc2023-results/{dset}_{d}{parti}_results.txt"
    with open(filename, 'r', encoding='utf8') as docfile:
        # header = docfile.readline()
        prop = read_next_property(dset, docfile, res_out[dset][d], rel_info, doc, ans)
        while prop:
            prop = read_next_property(dset, docfile, res_out[dset][d], rel_info, doc, ans)

    return res_out


def async_read_dev_results():
    res_out = {}
    dset = 'dev'
    num = 1000
    res_out[dset] = {}
    docred = read_docred(dset=dset, path=f"{working_dir}/data")
    with Pool(23) as p:
        jobs = [(d, doc) for d, doc in enumerate(docred)]
        for res in tqdm(p.imap(_worker, jobs), total=1000):
            for d in res[dset]:
                res_out[dset][d] = res[dset][d]
            del res
    return res_out


def repro_topk_parallel():
    res_out = async_read_dev_results()
    for k in range(1, 6):
        for restrict in [False, True]:
            results = top_k(res_out, k=k, restrict=restrict)
            hs = 0
            ms = 0
            for p, s in results['dev'].items():
                if sum(s) > 0:
                    print(f"{p}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")
                hs += s[0]
                ms += s[1]
            s = [hs, ms]

            print(f"total for {k}{'+dr' if restrict else ''}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")
    # tq.close()


def _worker2(args):
    prop, vals = args
    best = max(prf_all(vals).items(), key=lambda x: x[1]['f1'])
    return prop, best


def _worker3(args):
    prop, vals = args
    scores = prf_all(vals, restrict=True)
    best = max(scores.items(), key=lambda x: x[1]['f1'])
    return prop, best


def _workerp1366(args):
    prop, vals = args
    if prop != 'P1366':
        return None, None
    with open(f"{working_dir}/res/metrics/valsfilep1366.txt", 'w', encoding='utf8') as valsfile:
        valsfile.write(str(vals))

    scores = prf_all(vals, restrict=True)
    with open(f"{working_dir}/res/metrics/scoresfilep1366.txt", 'w', encoding='utf8') as scoresfile:
        scoresfile.write(str(scores))
    best = max(scores.items(), key=lambda x: x[1]['f1'])
    return prop, best


def p1366():
    res_out = async_read_dev_results()
    dset = 'dev'
    flatted = flatten(results=res_out)[dset]
    prop_jobs = [(prop, vals) for prop, vals in flatted.items()]

    with Pool(23) as p:
        for prop, best in list(tqdm(p.imap(_workerp1366, prop_jobs), total=96)):
            if prop:
                print(f"{prop}\t{best}\n")


def repro_prf():
    res_out = async_read_dev_results()
    dset = 'dev'
    flatted = flatten(results=res_out)[dset]
    top10 = ['P17', 'P27', 'P131', 'P150', 'P161', 'P175', 'P527', 'P569', 'P570', 'P577']
    prop_jobs = [(prop, vals) for prop, vals in flatted.items() if prop in top10]

    with Pool(10) as p:
        with open(f"{working_dir}/res/metrics/prf_dev_thresholds_all.txt", 'w', encoding='utf8') as out_file:
            for prop, best in list(tqdm(p.imap(_worker2, prop_jobs), total=96)):
                out_file.write(f"{prop}\t{best}\n")
        with open(f"{working_dir}/res/metrics/prf_dev_thresholds_dr.txt", 'w', encoding='utf8') as out_file:
            for prop, best in list(tqdm(p.imap(_worker3, prop_jobs), total=96)):
                out_file.write(f"{prop}\t{best}\n")


            # print(flatted)
            # print(prf_all(flatted))

            # for k in range(1, 6):
            #     for restrict in [False, True]:
            #
            #         results = top_k(res_out, k=k, restrict=restrict)
            #         hs = 0
            #         ms = 0
            #         for p, s in results['dev'].items():
            #             if sum(s) > 0:
            #                 print(f"{p}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")
            #             hs += s[0]
            #             ms += s[1]
            #         s = [hs, ms]
            #         print(f"total for {k}{'+dr' if restrict else ''}: {s} ({(s[0]/sum(s))*100.0:.2f}%)")

def main():
    # repro_topk_parallel()
    repro_prf()


if __name__ == '__main__':
    # tq = tqdm(total=1000)
    working_dir = "/data/git/text2kg2023-rilca-util"  # /res/eswc2023-results"
    participant = 0

    if participant == 0:
        rel_info = domain_range(get_rel_info(data_path=f"{working_dir}/data"))
    else:
        with open(f'{working_dir}/data/rel_info_participant_{participant}.json', 'r') as rel_info_file:
            rel_info = domain_range(json.load(rel_info_file))

    main()
    # print(domain_range(rel_info)['P17'])
    "?x was replaced by ?y in their role"
