import json
import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import pickle
from scipy.stats import pearsonr, ks_2samp, ttest_ind, spearmanr
from tqdm import tqdm
import collections
from multiprocessing import Pool
import os


exceptions = [23, 65, 67, 139, 344]
exception_rels = ['272', '162', '706']


# POOL_SIZE = 10


def pcorr(df, x, y, query=''):
    if query:
        df = df.query(query)

    df_x = df[x]
    df_y = df[y]
    if y == 'msd':
        df_y = 1-df[y]

    sz = len(df_x)
    assert sz == len(df_y), f"size mismatch? {sz}<>{len(df_y)}"

    # print(f"Query: {query}; Size: {sz}; Coefficient: {pearsonr(df_x, df_y)[0]}")
    return pearsonr(df_x, df_y)[0]
    # return


def scorr(df, x, y, query=''):
    if query:
        df = df.query(query)

    df_x = df[x]
    df_y = df[y]
    if y == 'msd':
        df_y = 1-df[y]

    sz = len(df_x)
    assert sz == len(df_y), f"size mismatch? {sz}<>{len(df_y)}"

    # print(f"Query: {query}; Size: {sz}; Coefficient: {pearsonr(df_x, df_y)[0]}")
    return spearmanr(df_x, df_y)
    # return


def ks_corr(x, y):
    return ks_2samp(x, y)


def _worker2(args):
    dir, model, i = args
    res = {}
    if i not in exceptions:
        metrics = ['pll', 'jsd', 'csd', 'esd', 'hsd', 'msd']
        for metric in metrics:
            with open(f"{dir}/{model}dev_{i}_{metric}s.csv", 'r', encoding='utf8') as score_file:
                rel = ''
                for j, line in enumerate(score_file):
                    parts = line.split('.\",')
                    if len(parts) > 1:
                        if rel not in exception_rels:
                            sent = hash(parts[0][1:])
                            agg, *scores = [float(s.strip()) for s in parts[1].strip().split(', ')]
                            # try:
                            #     vloc = res[sent]['v_loc']
                            # except KeyError as ke:
                            #     print(f"KEY ERROR: {i} {j} {rel} {parts[0][1:]}@{sent}")
                            #     raise ke
                            # if res[sent]['selected']:
                            if sent not in res:
                                res[sent] = {}
                            res[sent][metric] = {
                                'all': scores,
                                'agg': agg,
                                'avg': sum(scores)/len(scores),
                                # 'vrb': scores[vloc],
                            }
                    elif 'P' in line:
                        rel = line.strip().split('P')[-1]
    return res


def pickle_results(dir, max_doc=1000, model=""):

    if model and model[-1] != '_':
        model += "_"

    pickle_file = f'{dir}/pickles/{model or "bert-large-uncased_"}allscores.pickle'
    if os.path.exists(pickle_file):
        token_pickle_file = pickle_file.replace("allscores", "alltokeninfo")
        with open(pickle_file, 'rb') as handle, open(token_pickle_file, 'rb') as handle2:
            print(f"Found {pickle_file}, opening")
            # return
            return pickle.load(handle2), pickle.load(handle)

    print("Fetching token-level information...")
    res = fetch_token_info(dir=dir, max_doc=max_doc, model=model)
    print(len(res))
    print("Fetching metric values...")

    ret = {}
    with Pool(20) as p:
        jobs = [(dir, model, i) for i in range(max_doc)]
        for d, doc in enumerate(tqdm(p.imap(_worker2, jobs), total=max_doc)):
            # print("retlen:", len(ret))
            # print("doclen:", len(doc))
            for s, sent in enumerate(doc):
                if sent in res:
                    assert res[sent]['selected'], "Unselected?"
                    ret[sent] = doc[sent]
                    # vloc = res[sent]['v_loc']
                    # for metric in ret[sent]:
                    #     try:
                    #         ret[sent][metric]['vrb']
                    #     except:
                    #         print("Error 1 at ", model, d, metric, vloc)
                    #     try:
                    #         ret[sent][metric]['all'][vloc]
                    #     except:
                    #         print("Error 2 at ", model, d, metric, vloc)
                    #     ret[sent][metric]['vrb'] = ret[sent][metric]['all'][vloc]
    print("Starting pickling...")
    with open(pickle_file, 'wb') as handle:
        pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res, ret


def _worker(args):
    dir, model, i = args
    ret = {}
    if i not in exceptions:
        random.seed(4423 + i)
        with open(f"{dir}/{model}dev_{i}_tokens.csv", 'r', encoding='utf8') as score_file:
            rel = ''
            for j, line in enumerate(score_file):
                parts = line.split('.\",')
                if len(parts) > 1:
                    if rel not in exception_rels:
                        # "The head of government of Australia is NCOs.", False, 6, False, O, O, O, O, O, B, O, B, I, O
                        # 0: Sentence, 1: rest
                        sent = hash(parts[0][1:])
                        # if repeat:
                        #     if parts[0][1:] in back_hash[sent]:
                        #         assert f" {i}" not in back_hash[sent].split(":")[-1], f"Ugh, check {i} in  {back_hash[sent]}"
                        #         back_hash[sent] += f" {i}"
                        #         # print(f"Repeat: {back_hash[sent]}")
                        #     else:
                        #         print(f"{parts[0][1:]} {i} >><< {back_hash[sent]}")
                        #         raise Exception(f"{parts[0][1:]} {i} >><< {back_hash[sent]}")
                        # else:
                        #     back_hash[sent] = f"{parts[0][1:]}: {i}"
                        allwd, verb, corr, *scores = parts[1].strip().split(', ')
                        corr = corr == 'True'
                        allwd = allwd == 'True'
                        bio = list(s.strip() for s in scores)
                        repeat = sent in ret
                        if not repeat:
                            ret[sent] = {
                                'v_loc': int(verb),
                                'correct': corr,
                                'selected': corr or (random.random() > 0.99),
                                'bio': ''.join(bio),
                                # 'correct_bio': list(corr for _ in scores),
                                'allowed_dr': allwd,
                                # 'correct_allowed': list(allwd for _ in scores),
                                'both': (corr and allwd),
                                'mte': 'I' in bio
                            }
                        else:
                            ret[sent]['correct'] = ret[sent]['correct'] or corr
                            ret[sent]['selected'] = corr or ret[sent]['selected']
                            # ret[sent]['correct_bio'] = list(ret[sent]['correct'] for _ in scores)
                            ret[sent]['allowed_dr'] = ret[sent]['allowed_dr'] or allwd
                            # ret[sent]['correct_allowed'] = list(ret[sent]['allowed_dr'] for _ in scores)
                            ret[sent]['both'] = ret[sent]['both'] or (corr and allwd)
                elif 'P' in line:
                    rel = line.strip().split('P')[-1]
    return ret


def fetch_token_info(dir, max_doc=1000, model=""):
    random.seed(4423)
    pickle_file = f'{dir}/pickles/{model or "bert-large-uncased_"}alltokeninfo.pickle'
    # if os.path.exists(pickle_file):
    #     with open(pickle_file, 'rb') as handle:
    #         print(f"Found {pickle_file}, opening")
    #         return pickle.load(handle)


    ret = {}
    with Pool(40) as p:
        jobs = [(dir, model, i) for i in range(max_doc)]
        for doc in tqdm(p.imap(_worker, jobs), total=max_doc):
            for sent in doc:
                repeat = sent in ret
                if not repeat:
                    ret[sent] = doc[sent]
                else:
                    ret[sent]['correct'] = ret[sent]['correct'] or doc[sent]['correct']
                    ret[sent]['selected'] = ret[sent]['selected'] or doc[sent]['selected']
                    ret[sent]['allowed_dr'] = ret[sent]['allowed_dr'] or doc[sent]['allowed_dr']
                    # ret[sent]['correct_allowed'] = list(ret[sent]['allowed_dr'] for _ in scores)
                    ret[sent]['both'] = ret[sent]['both'] or doc[sent]['both']

        print(f"Dumping {pickle_file}")
        with open(pickle_file, 'wb') as handle:
            ret = {k: v for k, v in ret.items() if v['selected']}
            pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ret


def get_df(token_data, score_data, ignore=[]):
    mapping = {"agg":{}, "all":{}, "avg":{}}
    for sent in score_data:
        for metric in score_data[sent]:
            if ignore and metric in ignore:
                continue
            else:
                for m in mapping:
                    mapping[m][metric] = []
        break

    for m in mapping:
        mapping[m]['trues'] = []  # True/False statements
        mapping[m]['bios'] = []  # BIO tags
        mapping[m]['allows'] = []  # Restrictions
        mapping[m]['mtes'] = []  # Multi-token entity, is there one in the sentence?

    for sent in score_data:
        if not token_data[sent]['selected']:
            continue

        _ln = 0
        for metric in score_data[sent]:
            mapping['agg'][metric].append(score_data[sent][metric]['agg'])
            alls = score_data[sent][metric]['all']
            _ln = len(alls)
            mapping['avg'][metric].append(sum(alls)/_ln)
            mapping['all'][metric].extend(alls)

        for m in mapping:
            if m == 'all':
                mapping[m]['trues'].extend([token_data[sent]['correct']] * _ln)
                mapping[m]['allows'].extend([token_data[sent]['allowed_dr']] * _ln)
                mapping[m]['bios'].extend(list(token_data[sent]['bio']))
                mapping[m]['mtes'].extend([token_data[sent]['mte']] * _ln)
            else:
                mapping[m]['trues'].append(token_data[sent]['correct'])
                mapping[m]['allows'].append(token_data[sent]['allowed_dr'])
                mapping[m]['bios'].append(None)
                mapping[m]['mtes'].append(token_data[sent]['mte'])



    # for x in mapping_all:
    #     print(x, len(mapping_all[x]))

    # exit(0)
    df_all: pd.DataFrame = pd.DataFrame(mapping['all'])
    df_avg: pd.DataFrame = pd.DataFrame(mapping['avg'])
    df_agg: pd.DataFrame = pd.DataFrame(mapping['agg'])
    df_all.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    df_avg.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    df_agg.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    return df_all, df_avg, df_agg


def plot_pppl(df, ax, x='pll', label="PLL", model="unk", minp=-1000.0, maxp=1000.0, trues_allows=False, trues_mtes=False, only_true=False, only_allow=False, show=True):
    # fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    clp = (max(-1000.0, minp), min(maxp, 1000.0))
    plt.rcParams['hatch.linewidth'] = 0.3

    suffix = ''

    if only_allow:
        df = df.loc[df['allows']]
        suffix = ' Accepted'

    if only_true and (trues_allows or trues_mtes):
        df = df.loc[df['trues']]

    taas = ['Supported' + suffix if t else 'Unsupported' + suffix for t in df['trues']]

    if trues_allows:  # Show off four-element plot along two dimensions
        labelmap = {
            (True, True): 'Support+Accept',
            (True, False): 'Support+Reject',
            (False, True): 'Unsupp+Accept',
            (False, False): 'Unsupp+Reject',
        }

        taas = [labelmap[t] for t in df[['trues', 'allows']].apply(tuple, axis=1)]
        # ctr = collections.Counter(taas)
        # taas = [f'{t} ({ctr[t]})' for t in taas]
        # kde = sns.kdeplot(ax=ax, data=df, x="pll", hue=taas, common_norm=False, fill=True, clip=clp)

    # # hue=taas
    elif trues_mtes:
        labelmap = {
            (True, True): 'Supp w/MTEs',
            (True, False): 'Supp w/o MTEs',
            (False, True): 'Unsupp w/ MTEs',
            (False, False): 'Unsupp w/o MTEs',
        }
        # if only_allow:
        #     df = df.loc[df['allows']]

        taas = [labelmap[t] for t in df[['trues', 'mtes']].apply(tuple, axis=1)]
        # kde = sns.kdeplot(ax=ax, data=df, x="pll", hue=taas, common_norm=False, fill=True, clip=clp)
    # sns.move_legend(kde, "upper left")
    _outlabels = []
    for t in taas:
        if t not in _outlabels:
            _outlabels.append(t)
    ctr = collections.Counter(taas)
    # taas = [f'{t} ({ctr[t]})' for t in taas]
    taas = [f'{ctr[t]}' for t in taas]
    a = 3
    b = 3
    if len(ctr) == 2:
        hatches = ['\\' * a, '/' * a]
    else:
        hatches = ['\\' * a, '|' * b, '/' * a, '-' * b]

    kde = sns.kdeplot(ax=ax, data=df, x=x, hue=taas, common_norm=False, fill=True, clip=clp, hatch=hatches, alpha=0.4)
    for h, hatch in zip(ax.legend_.legend_handles, hatches):
        # h.set_label(f'{region}, {age}')
        h.set_hatch(hatch*2)
        # h.set_lc='w'
        # handles.append(h)
    for collection, hatch in zip(ax.collections[::-1], hatches):
        collection.set_hatch(hatch)
        # collection.set_edgecolor('w')
    if label == "PLL":
        kde.axis(xmin=-10.25, xmax=0.25, ymin=0, ymax=0.45)
        kde.xaxis.set_ticks(np.arange(-10.0, 0.1, 2.0))
        # kde.ylim([0, 0.5])
    elif label == "AVG_PL":
        # plt.xlim([-10.25, 0.25])
        # kde.axis(ymin=0, ymax=0.55)
        kde.axis(ymin=0, ymax=3.0)
        kde.xaxis.set_ticks(np.arange(0.0, 1.1, .2))
        # kde.ylim([0, 0.5])
    else:
        kde.axis(ymin=0, ymax=3.0)
        kde.xaxis.set_ticks(np.arange(0.0, 1.1, .2))
        # kde.ylim([0, 5.0])

    # snsplot: sns.JointGrid = sns.jointplot(data=df, x=a, y=b, hue='cats', alpha=alphas)
    # snsplot: sns.JointGrid = sns.jointplot(data=df.loc[df['cats'] == 'B'], x=a, y=b, hue='cats', alpha=0.05)

    sns.move_legend(
        kde, "upper left",
        # bbox_to_anchor=(.5, 1),
        ncol=2,
        # title='Counts:',
        # title_fontsize=6,
        frameon=False,
    )

    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Euclid Sim aggregate" if aggregate else b)


    # sns.move_legend(snsplot.ax_joint, "upper left", bbox_to_anchor=(1, 1.25))
    # snsplot = sns.jointplot(data=df, x=a, y=b, hue='cats', hue_order=['O', 'I', 'B'], alpha=0.1, legend=False)
    # snsplot = sns.jointplot(data=df, x=a, y=b, alpha=0.05, legend=False)
    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Euclid Sim aggregate" if aggregate else b)


    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Cosine Sim aggregate" if aggregate else b)

    # a: str, b: str, aggregate, verb_only=False, max_doc=100, dir="res", average=False, comp=False, density=False

    # exit(0)
    title_map_model = {
        "bert-large-cased": "BERT large",
        "bert-large-uncased": "BERT large uncased",
        "bert-base-cased": "BERT base",
        "roberta-base": "RoBERTa base",
        "roberta-large": "RoBERTa large",
    }

    title_label_model = {
        "PLL": "Per-sentence PLLs",
        "PL": "Per-token Pseudo-Likelihoods",
        "AVG_PL": "Per-sentence Avg Pseudo-Likelihoods",
    }

    # ax.set(xlabel=title_label_model[label])
    kde.set_title(title_map_model[model])
    # plt.title(title_map_model[model])  # for {title_map_model[model]}")

    # plt.subplots_adjust(top=0.812, bottom=0.233, left=0.133, right=0.95, hspace=0.2, wspace=0.2)

    kde.set(xlabel='')
    kde.label_outer()

    # return kde, _outlabels


    return kde, _outlabels
    if show:
        plt.show()
    else:
        return kde, _outlabels
        plt.margins(tight=True)
        imgfile = (f"plots/0_{label}_dev_"
                   f"{'TO_' if only_true else ''}"
                   f"{'AO_' if only_allow else ''}"
                   f"{'TA_' if trues_allows else ''}"
                   f"{'TM_' if trues_mtes else ''}"
                   f"{model}.png"
                   )
        plt.savefig(imgfile)
        print("File saved to", imgfile)

        plt.close(None)


def plot_scatter(df, ax, x='pll', y='csd', x_label="", y_label="", model="unk", trues_allows=False, trues_mtes=False, only_true=False, only_allow=False, show=True):
    suffix = ''

    if only_allow:
        df = df.loc[df['allows']]
        suffix = ' Accepted'

    if only_true and (trues_allows or trues_mtes):
        df = df.loc[df['trues']]

    taas = ['Supported' + suffix if t else 'Unsupported' + suffix for t in df['trues']]

    if trues_allows:  # Show off four-element plot along two dimensions
        labelmap = {
            (True, True): 'Support+Accept',
            (True, False): 'Support+Reject',
            (False, True): 'Unsupp+Accept',
            (False, False): 'Unsupp+Reject',
        }

        taas = [labelmap[t] for t in df[['trues', 'allows']].apply(tuple, axis=1)]
        # ctr = collections.Counter(taas)
        # taas = [f'{t} ({ctr[t]})' for t in taas]
        # kde = sns.kdeplot(ax=ax, data=df, x="pll", hue=taas, common_norm=False, fill=True, clip=clp)

    # # hue=taas
    elif trues_mtes:
        labelmap = {
            (True, True): 'Supp w/ MTEs',
            (True, False): 'Supp w/o MTEs',
            (False, True): 'Unsupp w/ MTEs',
            (False, False): 'Unsupp w/o MTEs',
        }
        # if only_allow:
        #     df = df.loc[df['allows']]

        taas = [labelmap[t] for t in df[['trues', 'mtes']].apply(tuple, axis=1)]
        # kde = sns.kdeplot(ax=ax, data=df, x="pll", hue=taas, common_norm=False, fill=True, clip=clp)
    # sns.move_legend(kde, "upper left")
    _outlabels = []
    for t in taas:
        if t not in _outlabels:
            _outlabels.append(t)
    ctr = collections.Counter(taas)
    # taas = [f'{t} ({ctr[t]})' for t in taas]
    taas = [f'{ctr[t]}' for t in taas]
    # df.sort_values('trues', inplace=True)
    alphas = [0.05 if not _ else .1 for _ in df['trues']]
    snsplot = sns.scatterplot(data=df, x=x, y=y, hue=taas, alpha=0.05)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label} ({model})")
    plt.show()


def compare(data, a: str, b: str, aggregate, verb_only=False, max_doc=100, dir="res", average=False, comp=False, density=False):
    # a_scores = {}
    # a_agg_scores = {}
    # b_scores = {}
    # b_agg_scores = {}
    #
    # token_info = fetch_token_info(max_doc)
    #
    #
    #
    # for i in range(max_doc):
    #     if i in exceptions:
    #         continue
    #     with open(f"{dir}/dev_{i}_{b}s.csv", 'r', encoding='utf8') as score_file:
    #         for line in score_file:
    #             parts = line.split('.\",')
    #             if len(parts) > 1:
    #                 sent = parts[0][1:]
    #                 scores = parts[1].strip().split(', ')
    #             if aggregate and verb_only:
    #                 # _scores = list(float(s.strip()) for s in scores[1:])
    #                 vscore = list(float(s.strip()) for s in scores[1:])
    #                 vscore = sum(vscore)/len(vscore)
    #                 # vscore = float(scores[v_loc[sent] + 1])
    #
    #                 b_agg_scores[sent] = vscore
    #                 b_scores[sent] = [vscore]
    #             else:
    #                 b_agg_scores[sent] = float(scores[0])
    #                 b_scores[sent] = list(float(s.strip()) for s in scores[1:])
    #
    # # print(b_scores)
    # # print("=="*50)
    # # print(b_agg_scores)
    #
    # if a == b:
    #     a = 'avg_' + a


    xs = []
    ys = []
    cs = []  # True/False statements
    cbs = []  # BIO tags
    crs = []  # Restrictions
    mts = []  # Multi-token entity, is there one in the sentence?
    # boths = []

    # for sent in a_agg_scores:
    #     xs.append(a_agg_scores[sent])
    #     ys.append(b_agg_scores[sent])

    # for sent in a_scores:
    #     _xs = a_scores[sent]
    #     _ys = b_scores[sent]
    #
    #     assert len(_xs) == len(_ys), f"{len(_xs)} != {len(_ys)} @ {sent}"
    #
    #     xs.extend(a_scores[sent])
    #     ys.extend(b_scores[sent])
    #
    # snsplot = sns.jointplot(x=xs, y=ys, alpha=0.01)
    # snsplot.set_axis_labels(xlabel=a, ylabel=b)
    #
    # # plt.scatter(xs, ys, alpha=0.2)
    # plt.show()

    # Schema:
    # sent: metric:
    #               all: []
    #               agg: float
    #               avg: float
    #               vrb: float
    #        v_loc: int
    #        correct: bool
    #        selected: bool
    #        bio: []
    #        allowed_dr: bool
    #        both: bool
    #        mte: bool

    for sent in data:
        if not data[sent]['selected']:
            continue

        if aggregate:
            _xs = data[sent][a]['agg']
            _ys = data[sent][b]['agg']
            _cs = data[sent]['correct']
            _crs = data[sent]['allowed_dr']
            _mts = data[sent]['mte']
            # _b = both[sent]

            # assert len(_xs) == len(_ys), f"{len(_xs)} != {len(_ys)} @ {sent}"

            xs.append(_xs)
            ys.append(_ys)
            cs.append(_cs)
            crs.append(_crs)
            cbs.append(None)
            mts.append(_mts)
            # boths.append(_b)
        else:
            _xs = data[sent][a]['all']
            _ys = data[sent][b]['all']
            _ln = len(_xs)
            _cs = [data[sent]['correct']] * _ln
            _crs = [data[sent]['allowed_dr']] * _ln
            _cbs = list(data[sent]['bio'])
            _mts = [None] * _ln

            assert len(_xs) == len(_ys), f"{len(_xs)} != {len(_ys)} @ {sent}"
            assert len(_xs) == len(_cs), f"{len(_xs)} != {len(_cs)} @ {sent}"

            xs.extend(_xs)
            ys.extend(_ys)
            cs.extend(_cs)
            crs.extend(_crs)
            cbs.extend(_cbs)
            mts.extend(_mts)

    df: pd.DataFrame = pd.DataFrame({a: xs, b: ys, 'trues': cs, 'bios': cbs, 'allows': crs, 'mtes': mts})

    # if aggregate:
    #     df[a] -= df[a].min()
    #     df[b] -= df[b].min()
    #     df[a] /= df[a].max()
    #     df[b] /= df[b].max()

    if a is None:
        print("Welch's t-test")

        # print(f"{df['pll'].min()} { df['pll'].max()} ")
        # print(f"{df['jsd'].min()} { df['jsd'].max()} ")

        df1 = df.loc[~df['trues']]
        df2 = df.loc[df['trues']]
        print(f"Var PPPL False: {np.var(df1['pll'])}")
        print(f"Var PPPL True:  {np.var(df2['pll'])}")
        print(f"Var jsd False: {np.var(df1['jsd'])}")
        print(f"Var jsd True:  {np.var(df2['jsd'])}")

        print("PPPL t-test:")
        print(ttest_ind(df1['pll'], df2['pll']))
        print(ttest_ind(df1['pll'], df2['pll'], equal_var=False))
        print("=="*50)
        print("JSD t-test:")
        print(ttest_ind(df1['jsd'], df2['jsd']))
        print(ttest_ind(df1['jsd'], df2['jsd'], equal_var=False))

        exit(0)


    if a is None:
        # df: pd.DataFrame = pd.DataFrame({a: xs, b: ys, 'cats': cs})
        print(f"total samps: {len(df)}")
        print(f"total samps: {pcorr(df, a, b)}")
        if aggregate:
            print(f"True samps: {len(df.loc[df['trues']])}")
            print(f"False samps: {len(~df.loc[df['trues']])}")
            print(f"True samps: {pcorr(df.loc[df['trues']], a, b)}")
            print(f"False samps: {pcorr(~df.loc[df['trues']], a, b)}")
        else:
            print(f"B samps: {len(df.loc[df['bios'] == 'B'])}")
            print(f"I samps: {len(df.loc[df['bios'] == 'I'])}")
            print(f"O samps: {len(df.loc[df['bios'] == 'O'])}")
            print(f"True B samps:", len(df.query('bios == "B" and trues')))
            print(f"True I samps:", len(df.query('bios == "I" and trues')))
            print(f"True O samps:", len(df.query('bios == "O" and trues')))
            print(f"False B samps:", len(df.query('bios == "B" and not trues')))
            print(f"False I samps:", len(df.query('bios == "I" and not trues')))
            print(f"False O samps:", len(df.query('bios == "O" and not trues')))
            print(f"B samps: {pcorr(df.loc[df['bios'] == 'B'], a, b)}")
            print(f"I samps: {pcorr(df.loc[df['bios'] == 'I'], a, b)}")
            print(f"O samps: {pcorr(df.loc[df['bios'] == 'O'], a, b)}")
            print(f"True B samps:", pcorr(df.query('bios == "B" and trues'), a, b))
            print(f"True I samps:", pcorr(df.query('bios == "I" and trues'), a, b))
            print(f"True O samps:", pcorr(df.query('bios == "O" and trues'), a, b))
            print(f"False B samps:", pcorr(df.query('bios == "B" and not trues'), a, b))
            print(f"False I samps:", pcorr(df.query('bios == "I" and not trues'), a, b))
            print(f"False O samps:", pcorr(df.query('bios == "O" and not trues'), a, b))

    # df = df.loc[df['allows'] == "True"]
    if aggregate:

        if comp:
            df.sort_values('trues', inplace=True)
            alphas = [0.05 if not _ else .1 for _ in df['trues']]
            snsplot: sns.JointGrid = sns.jointplot(data=df, x=a, y=b, hue='trues', alpha=alphas, marginal_kws={'common_norm':False})

        elif density:
            fig, ax = plt.subplots(figsize=(5, 2.7))

            trues_allows = True
            trues_mtes = False
            only_true = False
            only_allow = True

            if only_allow:
                df = df.loc[df['allows']]

            if only_true:
                df = df.loc[df['trues']]

            if trues_allows:  # Show off four-element plot along two dimensions
                labelmap = {
                    (True, True): 'Supported Accepted',
                    (True, False): 'Supported Rejected',
                    (False, True): 'Unsupported Accepted',
                    (False, False): 'Unsupported Rejected',
                }

                taas = df[['trues', 'allows']].apply(tuple, axis=1)
                from collections import Counter
                taas = [labelmap[t] for t in taas]
                ctr = Counter(taas)
                taas = [f'{t} ({ctr[t]})' for t in taas]
                kde = sns.kdeplot(ax=ax, data=df, x=a, hue=taas, common_norm=False, fill=True)

            # # hue=taas
            elif trues_mtes:
                labelmap = {
                    (True, True): 'Supported with MTEs',
                    (True, False): 'Supported without MTEs',
                    (False, True): 'Unsupported with MTEs',
                    (False, False): 'Unsupported without MTEs',
                }
                if only_allow:
                    df = df.loc[df['allows']]

                taas = df[['trues', 'mtes']].apply(tuple, axis=1)
                from collections import Counter
                taas = [labelmap[t] for t in taas]
                ctr = Counter(taas)
                taas = [f'{t} ({ctr[t]})' for t in taas]
                kde = sns.kdeplot(ax=ax, data=df, x=a, hue=taas, common_norm=False, fill=True)
            else:
                kde = sns.kdeplot(ax=ax, data=df, x=a, hue='trues', common_norm=False, fill=True)
                # #This is for filtering only by those that pass DR restrictions
                # # kde = sns.kdeplot(ax=ax, data=df.loc[df['allows']], x=a, hue='trues', common_norm=False, fill=True, palette=["C1", "C2"])
                # sns.move_legend(kde, "upper left")
                sns.move_legend(
                    kde, "lower center",
                    bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
                )
                plt.xlabel('PPPL')
                plt.subplots_adjust(top=0.812, bottom=0.233, left=0.133, right=0.95, hspace=0.2, wspace=0.2)
            # print(ks_2samp(df.loc[df['trues']][a], df.loc[~df['trues']][a], alternative='less'))





        # assert 0.5 in alphas
        # g = sns.FacetGrid(df, col="trues", hue="trues", margin_titles=True)
        # g = sns.FacetGrid(df, hue="cats")
        # g = sns.lmplot(data=df, x=a, y=b, hue="trues", scatter_kws={"alpha": {"True": 0.5, "False": 0.001}}, line_kws={"alpha": 1.0}, palette={"True": "g", "False": "m"})

        # g.map(plt.scatter, a, b, alpha=0.1)
    else:
        # xs, ys, cs, cbs = zip(*sorted(zip(xs, ys, cs, cbs), key=lambda x: 0 if x[2] == 'O' else (1 if x[2] == 'I' else 2)))
        # xs, ys, cs, cbs = zip(*sorted(zip(xs, ys, cs, cbs), key=lambda x: 2 if x[3] else 0))
        # alphas = [0.05 if _ == "False" else .5 for _ in cbs]


        # # Main way to look at it:
        # df: pd.DataFrame = pd.DataFrame({a: xs, b: ys, 'cats': cs, 'cats2': cbs})
        # g = sns.FacetGrid(df, row="cats2", col="cats", hue="cats2", margin_titles=True)
        # g.map(sns.scatterplot, a, b, alpha=0.1)

        fig, ax = plt.subplots(figsize=(5, 2.7))
        kde = sns.kdeplot(ax=ax, data=df, x=a, hue='trues', common_norm=False, fill=True, clip=(0.0, 1.0))
        # sns.move_legend(kde, "upper left")
        sns.move_legend(kde, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)

        plt.xlabel('PLL')
        plt.subplots_adjust(top=0.812, bottom=0.233, left=0.133, right=0.95, hspace=0.2, wspace=0.2)


    # snsplot: sns.JointGrid = sns.jointplot(data=df, x=a, y=b, hue='cats', alpha=alphas)
    # snsplot: sns.JointGrid = sns.jointplot(data=df.loc[df['cats'] == 'B'], x=a, y=b, hue='cats', alpha=0.05)


    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Euclid Sim aggregate" if aggregate else b)


    # sns.move_legend(snsplot.ax_joint, "upper left", bbox_to_anchor=(1, 1.25))
    # snsplot = sns.jointplot(data=df, x=a, y=b, hue='cats', hue_order=['O', 'I', 'B'], alpha=0.1, legend=False)
    # snsplot = sns.jointplot(data=df, x=a, y=b, alpha=0.05, legend=False)
    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Euclid Sim aggregate" if aggregate else b)


    # snsplot.set_axis_labels(xlabel="PPPL" if aggregate else a, ylabel="Cosine Sim aggregate" if aggregate else b)

    # a: str, b: str, aggregate, verb_only=False, max_doc=100, dir="res", average=False, comp=False, density=False
    plt.show()
    imgfile = (f"plots/fig_"
              f"{a}_vs_{b}_"
              f"{'AG_' if aggregate else ''}"
              f"{'VO_' if verb_only else ''}"
              f"{'AV_' if average else ''}"
              f"{'CO_' if comp else ''}"
              f"{'DE_' if density else ''}"
              f"{max_doc}docs_"
              f"{int(time.time())}_"
              f".png")
    print("File saved to", imgfile)
    plt.savefig(imgfile)


def print_junk(dir, model=""):
    ret = {}
    with Pool(40) as p:
        jobs = [(dir, model, i) for i in range(1000)]
        for doc in tqdm(p.imap(_worker, jobs), total=1000):
            for sent in doc:
                repeat = sent in ret
                if not repeat:
                    ret[sent] = doc[sent]
                else:
                    ret[sent]['correct'] = ret[sent]['correct'] or doc[sent]['correct']
                    # ret[sent]['selected'] = ret[sent]['selected'] or doc[sent]['selected']
                    # ret[sent]['allowed_dr'] = ret[sent]['allowed_dr'] or doc[sent]['allowed_dr']
                    # # ret[sent]['correct_allowed'] = list(ret[sent]['allowed_dr'] for _ in scores)
                    # ret[sent]['both'] = ret[sent]['both'] or doc[sent]['both']

    supported = 0
    unsupported = 0
    for sent in ret:
        if ret[sent]['correct']:
            supported += 1
        else:
            unsupported += 1
    print(f"Supported statements: {supported}")
    print(f"Unsupported statements: {unsupported}")


def test_same_metric(df_agg, model:str, metric: str):
    # alltokens, allscores = pickle_results(dir=resdir, model=model)
    # df, df_avg, df_agg = get_df(alltokens, allscores, ignore=[])

    # print(df_agg['pll'])
    # df_agg_sup = df_avg[df_avg['trues']]
    # df_agg_uns = df_avg[~df_avg['trues']]
    # print(len(df_agg_sup))
    # print(len(df_agg_sup[df_agg_sup['pll'] >= 0.6]))
    # print(len(df_agg_sup[df_agg_sup['pll'] >= 0.8]))
    # print(len(df_agg_uns))
    # print(len(df_agg_uns[df_agg_uns['pll'] >= 0.6]))
    # print(len(df_agg_uns[df_agg_uns['pll'] >= 0.8]))

    # print(pearsonr(df_agg_sup[df_agg_sup['trues']], df_agg_uns[df_agg_uns['trues']])[0])
    # print(len(df_agg_sup[df_agg_sup['trues']]))
    # print(len(df_agg_sup[df_agg_sup['trues']]['pll']))
    # print(len(df_agg_uns[df_agg_uns['trues']]))
    # print(len(df_agg_uns[df_agg_uns['trues']]['pll']))
    df_agg_alw = df_agg[df_agg['allows']]
    df_agg_una = df_agg[~df_agg['allows']]
    print(model)
    #
    #
    #
    #
    #
    #
    print("Accepted_Supported", len(df_agg_alw[df_agg_alw['trues']][metric]))
    print("Rejected_Unsupported", len(df_agg_una[~df_agg_una['trues']][metric]))
    print("Accepted_Unsupported", len(df_agg_alw[~df_agg_alw['trues']][metric]))
    print("Rejected_Supported", len(df_agg_una[df_agg_una['trues']][metric]))

    print("Accepted_Supported/Rejected_Supported", ks_corr(df_agg_alw[df_agg_alw['trues']][metric], df_agg_una[df_agg_una['trues']][metric]))
    print("Accepted_Supported/Rejected_Unsupported", ks_corr(df_agg_alw[df_agg_alw['trues']][metric], df_agg_una[~df_agg_una['trues']][metric]))
    print("Accepted_Supported/Accepted_Unsupported", ks_corr(df_agg_alw[df_agg_alw['trues']][metric], df_agg_alw[~df_agg_alw['trues']][metric]))
    print("Rejected_Supported/Rejected_Unsupported", ks_corr(df_agg_una[df_agg_una['trues']][metric], df_agg_una[~df_agg_una['trues']][metric]))
    print("Rejected_Supported/Accepted_Unsupported", ks_corr(df_agg_una[df_agg_una['trues']][metric], df_agg_alw[~df_agg_alw['trues']][metric]))
    print("Rejected_Unsupported/Accepted_Unsupported", ks_corr(df_agg_una[~df_agg_una['trues']][metric], df_agg_alw[~df_agg_alw['trues']][metric]))
    
    df_agg_mte = df_agg_alw[df_agg_alw['mtes']]
    df_agg_ste = df_agg_alw[~df_agg_alw['mtes']]
    
    print("MTE_Supported", len(df_agg_mte[df_agg_mte['trues']][metric]))
    print("STE_Unsupported", len(df_agg_ste[~df_agg_ste['trues']][metric]))
    print("MTE_Unsupported", len(df_agg_mte[~df_agg_mte['trues']][metric]))
    print("STE_Supported", len(df_agg_ste[df_agg_ste['trues']][metric]))

    print("MTE_Supported/STE_Supported", ks_corr(df_agg_mte[df_agg_mte['trues']][metric], df_agg_ste[df_agg_ste['trues']][metric]))
    print("MTE_Supported/STE_Unsupported", ks_corr(df_agg_mte[df_agg_mte['trues']][metric], df_agg_ste[~df_agg_ste['trues']][metric]))
    print("MTE_Supported/MTE_Unsupported", ks_corr(df_agg_mte[df_agg_mte['trues']][metric], df_agg_mte[~df_agg_mte['trues']][metric]))
    print("STE_Supported/STE_Unsupported", ks_corr(df_agg_ste[df_agg_ste['trues']][metric], df_agg_ste[~df_agg_ste['trues']][metric]))
    print("STE_Supported/MTE_Unsupported", ks_corr(df_agg_ste[df_agg_ste['trues']][metric], df_agg_mte[~df_agg_mte['trues']][metric]))
    print("STE_Unsupported/MTE_Unsupported", ks_corr(df_agg_ste[~df_agg_ste['trues']][metric], df_agg_mte[~df_agg_mte['trues']][metric]))
    


def test_diff_metric(df, df_agg, model):
    # alltokens, allscores = pickle_results(dir=resdir, model=model)
    # df, df_avg, df_agg = get_df(alltokens, allscores, ignore=[])
    # df: pd.DataFrame = pd.DataFrame({a: xs, b: ys, 'cats': cs})

    print("Model:", model)
    # print(f"total samps: {len(df)}")
    # print(f"total samps: {pcorr(df, a, b)}")


    queries = [
        'trues',
        'not trues',

        'bios == "B"',
        'bios == "I"',
        'bios == "O"',

        'bios == "B" and trues',
        'bios == "I" and trues',
        'bios == "O" and trues',

        'bios == "B" and not trues',
        'bios == "I" and not trues',
        'bios == "O" and not trues',

        'trues and (bios == "B" or bios == "I")',
        'not trues and (bios == "B" or bios == "I")',
    ]

    queries_agg = [
        '',
        'trues',
        'not trues',

        'allows',
        'not allows',
        'allows and trues',  # 5
        'not allows and trues',  # 6
        'allows and not trues',  # 7
        'not allows and not trues',  # 8

        'allows and mtes',
        'allows and not mtes',
        'allows and mtes and trues',  # 11
        'allows and not mtes and trues',  # 12
        'allows and mtes and not trues',  # 13
        'allows and not mtes and not trues',  # 14
    ]

    mm = {'jsd':"JSD", 'csd':"CS", 'hsd':"HD", 'msd':"MSD"}

    for metric in ['csd', 'jsd', 'msd', 'hsd']:
        print()
        # print("Aggregate scores", metric)
        vals = []
        valsb = []
        for query in queries_agg:
            score = scorr(df_agg, 'pll', metric, query)
            vals.append(score[0])
            valsb.append(score[1])
        # vals = [vals[5], vals[6], vals[7], vals[8], vals[11], vals[12], vals[13], vals[14]]

        # which = [0, 1, 2]
        which = [5, 6, 7, 8, 11, 12, 13, 14]
        vals = [f"{vals[w]:.2f}, $p={valsb[w]:.3f}$" for w in which]
        # valsb = [valsb[w] for w in which]

        print(f"{mm[metric]} & ", " & ".join(vals), "\\\\")
        # print("Per-token scores:")
        for query in queries:
            continue
            pcorr(df, 'pll', metric, query)
    print("==" * 50)



    #
    # print(f"B samps: {len(df.loc[df['bios'] == 'B'])}")
    # print(f"I samps: {len(df.loc[df['bios'] == 'I'])}")
    # print(f"O samps: {len(df.loc[df['bios'] == 'O'])}")
    # print(f"True B samps:", len(df.query('bios == "B" and trues')))
    # print(f"True I samps:", len(df.query('bios == "I" and trues')))
    # print(f"True O samps:", len(df.query('bios == "O" and trues')))
    # print(f"False B samps:", len(df.query('bios == "B" and not trues')))
    # print(f"False I samps:", len(df.query('bios == "I" and not trues')))
    # print(f"False O samps:", len(df.query('bios == "O" and not trues')))
    # print(f"B samps: {pcorr(df.loc[df['bios'] == 'B'], a, b)}")
    # print(f"I samps: {pcorr(df.loc[df['bios'] == 'I'], a, b)}")
    # print(f"O samps: {pcorr(df.loc[df['bios'] == 'O'], a, b)}")
    # print(f"True B samps:", pcorr(df.query('bios == "B" and trues'), a, b))
    # print(f"True I samps:", pcorr(df.query('bios == "I" and trues'), a, b))
    # print(f"True O samps:", pcorr(df.query('bios == "O" and trues'), a, b))
    # print(f"False B samps:", pcorr(df.query('bios == "B" and not trues'), a, b))
    # print(f"False I samps:", pcorr(df.query('bios == "I" and not trues'), a, b))
    # print(f"False O samps:", pcorr(df.query('bios == "O" and not trues'), a, b))


if __name__ == '__main__':
    with open(f'data/rel_info_full.json', 'r') as rel_info_file:
        rel_info = json.load(rel_info_file)

    resdir = "/data/experiments/kcap-2023/res"
    # resdir = "/kcap-2023/res"

    # alltokens, allscores = pickle_results(dir=resdir, model='bert-base-cased')

    # print_junk(resdir, "bert-large-cased_")
    # exit(0)


    # resdir = sys.argv[1]
    #
    # for model in ["", "bert-large-cased", "bert-base-cased", "roberta-large", "roberta-base"]:
    #     # model = "bert-large-cased_"
    #     try:
    #         print("Trying", model)
    #     except Exception as e:
    #         print("Failed.")
    #         pass

    # exit(0)

    # ops = ['pll', 'jsd', 'csd', 'hsd', 'esd', 'msd']

    # for o1 in ops:
    #     for o2 in ops:
    #         if o1 != o2:
    #             compare('pll', 'jsd', aggregate=False, verb_only=False, density=False, max_doc=10, dir="/data/experiments/kcap-2023/res")
    #             exit(0)

    # with open('allscores.pickle', 'rb') as handle:

    # model = "bert-large-cased"

    # for model in ["bert-large-cased", "roberta-base", "bert-large-uncased", "bert-base-cased"]:
    # for model in ["roberta-large"]:

    dataframes = {}

    print("loading...")
    models = ["bert-large-cased", "bert-base-cased", "roberta-large", "roberta-base"]
    # models = ["roberta-base", "roberta-large"]
    # models = ["roberta-large"]
    for model in set(models):
        # continue
        print(model)
        alltokens, allscores = pickle_results(dir=resdir, model=model)
        df, df_avg, df_agg = get_df(alltokens, allscores, ignore=[])
        dataframes[model] = (df_agg, df, df_avg)
        dataframes[model] = (df_agg, df_avg)
        # test_same_metric(df_agg, model, 'pll')
        # test_diff_metric(df, df_agg, model)
        # ./kcap-2023/src/run.sh
    #      pip install matplotlib seaborn scipy && cd /kcap-2023/src/ &&
    #      PYTHONHASHSEED=4423 && python scripts/eval/plot_compare_metrics.py
    # exit(0)
    print("plotting...")


    # for aggregate in [True, False]:
    #     for verb in [True, False]:
    #         for density in [True, False]:
    #             pass

    kargs = \
        [
         {'trues_allows': False, 'trues_mtes': False, 'only_true': False, 'only_allow': False},
         {'trues_allows': False, 'trues_mtes': False, 'only_true': False, 'only_allow': True},
         {'trues_allows': True, 'trues_mtes': False, 'only_true': False, 'only_allow': False},
         # {'trues_allows': True, 'trues_mtes': False, 'only_true': True, 'only_allow': True},  # Not actually useful.
         {'trues_allows': False, 'trues_mtes': True, 'only_true': False, 'only_allow': False},
         {'trues_allows': False, 'trues_mtes': True, 'only_true': False, 'only_allow': True},
         {'trues_allows': False, 'trues_mtes': True, 'only_true': True, 'only_allow': True},
         ]
    # for kws in kargs:
    #     show = True
    #     plot_pppl(df_agg, 'PLL', maxp=0.0, model=model, show=show, **kws)
    #     plot_pppl(df, 'PL', minp=0.0, maxp=1.0, model=model, show=show, **kws)
    #     plot_pppl(df_avg, 'AVG_PL', minp=0.0, maxp=1.0, model=model, show=show, **kws)
    #     exit(0)
    #     if show:
    #         exit(0)
    # compare(allscores, 'pll', 'jsd', aggregate=True, verb_only=False, density=False, comp=True, max_doc=1000, dir=resdir)
    title_label_model = {
        "PLL": "Per-sentence ",
        "PL": "Per-token Pseudo-Likelihoods",
        "AVG_PL": "Per-sentence Average Pseudo-Likelihoods",
    }

    title_desc_map = {
        'pll': "PLLs",
        'jsd': "Jensen-Shannon divergence (JSD)",
        'csd': "Cosine Similarity (CS)",
        'hsd': "Hellinger distance (HD)",
        'msd': "Mean squared deviation (MSD)"
    }

    TINY_SIZE = 6
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=5)  # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

    # plot_scatter(dataframes["roberta-large"][1], None, x='pll', y='csd', x_label="PLL", y_label="CS", model="roberta-large", show=True, **kargs[0])
    # exit(0)

    for kws in kargs:
        show = True

        for x in ['pll', 'csd']:#, 'jsd', 'csd', 'hsd', 'msd']:
            for l, label in enumerate(['PLL', 'AVG_PL']):
            # for l, label in enumerate(['PLL']):
                sns.set_palette('colorblind')
                # sns.set_context('paper')
                fig, axes = plt.subplots(1, len(models), figsize=(7, 1.5), dpi=300, sharey='all', sharex='all')
                for m, model in enumerate(models):
                    if label == 'PLL':
                        ax, ol = plot_pppl(dataframes[model][l], axes[m], x=x, label=label, maxp=0.0, model=model, show=show, **kws)
                    else:
                        ax, ol = plot_pppl(dataframes[model][l], axes[m], x=x, label=label, minp=0.0, maxp=1.0, model=model, show=show, **kws)
                handles = ax.legend_.legend_handles
                # fig.subplots_adjust(
                #     top=0.750,
                #     bottom=0.137,
                #     left=0.062,
                #     right=0.869,
                #     hspace=0.16,
                #     wspace=0.08
                # )

                fig.subplots_adjust(
                    top=0.98,
                    bottom=0.250,
                    left=0.062,
                    right=0.869,
                    hspace=0.16,
                    wspace=0.08
                )

                # fig.suptitle(title_label_model[label] + title_desc_map[x])
                info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.1,
                        'ha': 'center', 'va': 'top', 'rotation': 0,
                        'size': 'figure.titlesize', 'weight': 'figure.titleweight'}
                fig._suplabels(title_label_model[label] + title_desc_map[x], info)

                # plt.legend(handles=handles, labels=ol, bbox_to_anchor=(1.05, 1), loc=2, ncol=1)
                plt.figlegend(handles=handles, labels=ol, loc=7, ncol=1)

                # plt.margins(tight=True)
                # plt.show()
                # continue
                imgfile = (f"plots/0_{x}_{label}_dev_"
                           f"{'TO_' if kws['only_true'] else ''}"
                           f"{'AO_' if kws['only_allow'] else ''}"
                           f"{'TA_' if kws['trues_allows'] else ''}"
                           f"{'TM_' if kws['trues_mtes'] else ''}"
                           f".png"
                           )
                plt.savefig(imgfile)
                print("File saved to", imgfile)

                plt.close(None)
    # compare('pll', 'rand_normal', aggregate=False, max_doc=20)


# categories that I'd like:
# In/out of entity (Check)
# Multi-word entity (check)
# Correct/incorrect statements (check)
#
# In/out of domain (check)




