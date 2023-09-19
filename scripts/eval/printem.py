import json
if __name__ == '__main__':
    # with open('/data/git/text2kg2023-rilca-util/res/metrics/prf_dev_thresholds_all.txt') as res_file:
    #     # P108(-1.05, {'tp': 5, 'fp': 61, 'fn': 91, 'precision': 0.07575757575757576, 'recall': 0.052083333333333336, 'f1': 0.0308641975308642})
    #     for line in res_file:
    #         p, rest = line.split('\t', 1)
    #         if p in ['P17','P27','P131','P150','P161','P175','P527','P569','P570','P577']:
    #             rest = rest[1:-2]
    #             t, js = rest.split(',', 1)
    #             # print()
    #             js = json.loads(js.replace("'", '"'))
    #             print(f"{p}  & {t} & {js['f1']:.2f} & {js['precision']:.2f} & {js['recall']:.2f} & {js['tp']} & {js['fp']} & {js['fn']}\\\\")
    #
    # with open('/data/git/text2kg2023-rilca-util/res/metrics/prf_dev_thresholds_dr.txt') as res_file:
    #     # P108(-1.05, {'tp': 5, 'fp': 61, 'fn': 91, 'precision': 0.07575757575757576, 'recall': 0.052083333333333336, 'f1': 0.0308641975308642})
    #     for line in res_file:
    #         p, rest = line.split('\t', 1)
    #         if p in ['P17','P27','P131','P150','P161','P175','P527','P569','P570','P577']:
    #             rest = rest[1:-2]
    #             t, js = rest.split(',', 1)
    #             # print()
    #             js = json.loads(js.replace("'", '"'))
    #             print(f"{p}  & {t} & {js['f1']:.2f} & {js['precision']:.2f} & {js['recall']:.2f} & {js['tp']} & {js['fp']} & {js['fn']}\\\\")

    for i in range(4):
        with open(f'/data/git/text2kg2023-rilca-util/res/metrics/topkat1000detailed_P{i}.txt') as res_file:
            # P108(-1.05, {'tp': 5, 'fp': 61, 'fn': 91, 'precision': 0.07575757575757576, 'recall': 0.052083333333333336, 'f1': 0.0308641975308642})
            top10 = ['P17','P27','P131','P150','P161','P175','P527','P569','P570','P577']
            res = {p: [] for p in top10}
            for i, line in enumerate(res_file):
                if i < 2:
                    continue
                if len(line) == 0:
                    continue
                if line[0] != 'P':
                    continue
                # P17: [178, 408](30.38 %)
                p, rest = line.split(':', 1)
                if p in top10:
                    rest = rest[1:-4]
                    _, js = rest.split('(', 1)
                    res[p].append(js)
                    # print()
            for p, r in res.items():
                # print(f"{p}  & {r[0]} & {r[2]} & {r[4]} & {r[6]} & {r[8]}\\\\")
                # print(f"     & {r[1]} & {r[3]} & {r[5]} & {r[7]} & {r[9]}\\\\")

                print(f" & {r[0]} & {r[8]}")
                print(f" & {r[1]} & {r[9]}")
        print("=="*30)

