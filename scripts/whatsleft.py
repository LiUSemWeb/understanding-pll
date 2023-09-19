import time
import os
mlms = ["bert-large-uncased", "bert-large-cased", "bert-base-cased", "roberta-large", "roberta-base"]
metrics = ['token', 'pll', 'jsd', 'csd', 'hsd', 'esd', 'msd']
docs = 1000

last_totals = []


def printit(dfs):
    os.system('clear')
    total = 0
    for mlm in mlms:
        for metric in metrics:
            total += len(dfs[mlm][metric])

    print("TOTAL REMAINING:", total)
    print("\t\t".join(f"{mlm: <16}" for mlm in mlms))
    for metric in metrics:
        print("\t\t".join(f"  {metric+':': <9}{len(dfs[mlm][metric]): 5}" for mlm in mlms))


if __name__ == '__main__':
    dfs = {}
    for mlm in mlms:
        dfs[mlm] = {}
        for metric in metrics:
            dfs[mlm][metric] = list(range(docs))

    while len(dfs) > 0:
        for f in os.listdir('/data/experiments/kcap-2023/res'):
            parts = f[:-5].split('_')
            if len(parts) == 1:
                continue
            elif len(parts) == 3:
                mlm = "bert-large-uncased"
            else:
                mlm = parts[0]
            d = int(parts[-2])
            metric = parts[-1]
            # print((d, mlm, metric))
            if d in dfs[mlm][metric]:
                dfs[mlm][metric].remove(d)
        printit(dfs)
        time.sleep(15)

