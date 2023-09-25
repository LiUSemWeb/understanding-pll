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


def pcorr(df, x, y, query=''):
    if query:
        df = df.query(query)

    df_x = df[x]
    df_y = df[y]
    if y == 'msd':
        df_y = 1-df[y]

    sz = len(df_x)
    assert sz == len(df_y), f"size mismatch? {sz}<>{len(df_y)}"

    return pearsonr(df_x, df_y)[0]


def scorr(df, x, y, query=''):
    if query:
        df = df.query(query)

    df_x = df[x]
    df_y = df[y]
    if y == 'msd':
        df_y = 1-df[y]

    sz = len(df_x)
    assert sz == len(df_y), f"size mismatch? {sz}<>{len(df_y)}"

    return spearmanr(df_x, df_y)


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
                            if sent not in res:
                                res[sent] = {}
                            res[sent][metric] = {
                                'all': scores,
                                'agg': agg,
                                'avg': sum(scores)/len(scores),
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
            return pickle.load(handle2), pickle.load(handle)

    print("Fetching token-level information...")
    res = fetch_token_info(dir=dir, max_doc=max_doc, model=model)
    print(len(res))
    print("Fetching metric values...")

    ret = {}
    with Pool(20) as p:
        jobs = [(dir, model, i) for i in range(max_doc)]
        for d, doc in enumerate(tqdm(p.imap(_worker2, jobs), total=max_doc)):
            for s, sent in enumerate(doc):
                if sent in res:
                    assert res[sent]['selected'], "Unselected?"
                    ret[sent] = doc[sent]
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
                        sent = hash(parts[0][1:])
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
    df_all: pd.DataFrame = pd.DataFrame(mapping['all'])
    df_avg: pd.DataFrame = pd.DataFrame(mapping['avg'])
    df_agg: pd.DataFrame = pd.DataFrame(mapping['agg'])
    df_all.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    df_avg.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    df_agg.sort_values(['trues', 'allows', 'mtes'], inplace=True)
    return df_all, df_avg, df_agg


def plot_pppl(df, ax, x='pll', label="PLL", model="unk", minp=-1000.0, maxp=1000.0, trues_allows=False, trues_mtes=False, only_true=False, only_allow=False, show=True):
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
    elif trues_mtes:
        labelmap = {
            (True, True): 'Supp w/MTEs',
            (True, False): 'Supp w/o MTEs',
            (False, True): 'Unsupp w/ MTEs',
            (False, False): 'Unsupp w/o MTEs',
        }

        taas = [labelmap[t] for t in df[['trues', 'mtes']].apply(tuple, axis=1)]
    _outlabels = []
    for t in taas:
        if t not in _outlabels:
            _outlabels.append(t)
    ctr = collections.Counter(taas)
    taas = [f'{ctr[t]}' for t in taas]
    a = 3
    b = 3
    if len(ctr) == 2:
        hatches = ['\\' * a, '/' * a]
    else:
        hatches = ['\\' * a, '|' * b, '/' * a, '-' * b]

    kde = sns.kdeplot(ax=ax, data=df, x=x, hue=taas, common_norm=False, fill=True, clip=clp, hatch=hatches, alpha=0.4)
    for h, hatch in zip(ax.legend_.legend_handles, hatches):
        h.set_hatch(hatch*2)
    for collection, hatch in zip(ax.collections[::-1], hatches):
        collection.set_hatch(hatch)
    if label == "PLL":
        kde.axis(xmin=-10.25, xmax=0.25, ymin=0, ymax=0.45)
        kde.xaxis.set_ticks(np.arange(-10.0, 0.1, 2.0))
    elif label == "AVG_PL":
        kde.axis(ymin=0, ymax=3.0)
        kde.xaxis.set_ticks(np.arange(0.0, 1.1, .2))
    else:
        kde.axis(ymin=0, ymax=3.0)
        kde.xaxis.set_ticks(np.arange(0.0, 1.1, .2))

    sns.move_legend(
        kde, "upper left",
        ncol=2,
        frameon=False,
    )

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

    kde.set_title(title_map_model[model])
    kde.set(xlabel='')
    kde.label_outer()

    return kde, _outlabels
    # if show:
    #     plt.show()
    # else:
    #     return kde, _outlabels
    #     plt.margins(tight=True)
    #     imgfile = (f"plots/0_{label}_dev_"
    #                f"{'TO_' if only_true else ''}"
    #                f"{'AO_' if only_allow else ''}"
    #                f"{'TA_' if trues_allows else ''}"
    #                f"{'TM_' if trues_mtes else ''}"
    #                f"{model}.png"
    #                )
    #     plt.savefig(imgfile)
    #     print("File saved to", imgfile)
    #
    #     plt.close(None)


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
    elif trues_mtes:
        labelmap = {
            (True, True): 'Supp w/ MTEs',
            (True, False): 'Supp w/o MTEs',
            (False, True): 'Unsupp w/ MTEs',
            (False, False): 'Unsupp w/o MTEs',
        }

        taas = [labelmap[t] for t in df[['trues', 'mtes']].apply(tuple, axis=1)]
    _outlabels = []
    for t in taas:
        if t not in _outlabels:
            _outlabels.append(t)
    ctr = collections.Counter(taas)
    taas = [f'{ctr[t]}' for t in taas]
    alphas = [0.05 if not _ else .1 for _ in df['trues']]
    snsplot = sns.scatterplot(data=df, x=x, y=y, hue=taas, alpha=0.05)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label} ({model})")
    plt.show()


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

    df_agg_alw = df_agg[df_agg['allows']]
    df_agg_una = df_agg[~df_agg['allows']]
    print(f"Two-sample Kolmogorov-Smirnov tests for goodness of fit: {model}")

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
    print("Model:", model)
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

    mm = {'jsd': "JSD", 'csd': "CS", 'hsd': "HD", 'msd': "MSD"}

    for metric in ['csd', 'jsd', 'msd', 'hsd']:
        print()
        vals = []
        valsb = []
        for query in queries_agg:
            score = scorr(df_agg, 'pll', metric, query)
            vals.append(score[0])
            valsb.append(score[1])
        # which = [0, 1, 2]
        which = [0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14]
        vals = [f"{vals[w]:.2f}, $p={valsb[w]:.3f}$" for w in which]
        # valsb = [valsb[w] for w in which]

        print(f"{mm[metric]} & ", " & ".join(vals), "\\\\")
        # print("Per-token scores:")
        for query in queries:
            pcorr(df, 'pll', metric, query)
    print("==" * 50)


if __name__ == '__main__':
    with open(f'data/rel_info_full.json', 'r') as rel_info_file:
        rel_info = json.load(rel_info_file)

    resdir = "res"
    dataframes = {}

    print("loading...")
    models = ["bert-large-cased", "bert-base-cased", "roberta-large", "roberta-base"]
    for model in models:
        # continue
        print(model)
        alltokens, allscores = pickle_results(dir=resdir, model=model)
        df, df_avg, df_agg = get_df(alltokens, allscores, ignore=[])
        dataframes[model] = (df_agg, df, df_avg)
        dataframes[model] = (df_agg, df_avg)
        test_same_metric(df_agg, model, 'pll')
        test_diff_metric(df, df_agg, model)
    print("plotting...")

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

    for kws in kargs:
        show = True

        for x in ['pll', 'jsd', 'csd', 'hsd', 'msd']:
            for l, label in enumerate(['PLL', 'AVG_PL']):
                sns.set_palette('colorblind')
                fig, axes = plt.subplots(1, len(models), figsize=(7, 1.5), dpi=300, sharey='all', sharex='all')
                for m, model in enumerate(models):
                    if label == 'PLL':
                        ax, ol = plot_pppl(dataframes[model][l], axes[m], x=x, label=label, maxp=0.0, model=model, show=show, **kws)
                    else:
                        ax, ol = plot_pppl(dataframes[model][l], axes[m], x=x, label=label, minp=0.0, maxp=1.0, model=model, show=show, **kws)
                handles = ax.legend_.legend_handles

                fig.subplots_adjust(
                    top=0.98,
                    bottom=0.250,
                    left=0.062,
                    right=0.869,
                    hspace=0.16,
                    wspace=0.08
                )

                info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.1,
                        'ha': 'center', 'va': 'top', 'rotation': 0,
                        'size': 'figure.titlesize', 'weight': 'figure.titleweight'}
                fig._suplabels(title_label_model[label] + title_desc_map[x], info)

                plt.figlegend(handles=handles, labels=ol, loc=7, ncol=1)

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
