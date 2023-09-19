# Understanding and Estimating PLL
Code, data, and augmentation scripts for the IJCKG 2023 submission "Understanding and Estimating Pseudo-Log-Likelihood for Zero-Shot Fact Extraction with Masked Language Models."

If this message is here, it means that this README is incomplete.
The code should all be there.


## Data
You will need the original DocRED files.
The instructions for that can be found here: https://github.com/thunlp/DocRED
Place `dev.json` in `data/` and optionally include `train_annotated.json`.
We have already included an extended version of `rel_info.json` there.
It includes the prompts that were used in the experiments along with other annotations, some of which are unused and may be incorrect placeholders.


## Docker
Docker is not necessary to run this project, we simply used docker containers to parallelize the experiments across the machines available to us.
`build.sh`, `run.sh`, `run_exp.sh`, and the `images/` folder are all there to support that infrastructure.
However, the underlying python scripts can be run on any CUDA-enabled machine.


## Experiments
To run the bulk of the experiments, run the following from the root directory:
`python scripts/exp.py dev data res 512 256`

Here, `dev` is the data set (alternative would be `train`, but that was not used in the paper), `data` is the folder where the data is stored, `res` is the folder where output should be saved, 512 is the batch size for number of statements to consider at one time, and 256 is the batch size for the forward pass through the MLM.
These last two are optional and can be adjusted depending on if there are memory issues and what kind those are.
This is a very slow process, and is helped tremendously by running the code on multiple machines and pointing `res` to a shared folder.
To monitor progress, `whatsleft.py` can be used, but has no other purpose than to be informative.


To analyze the results, use:
`python scripts/eval/plot_compare_metrics.py res`

Here, `res` points to the folder where the results are stored.
It will first read all of the results and filter unnecessary data, then save the filtered data to a few pickle files.
Otherwise, this exhausts the memory of machines with only 128GB available.
The script will output various metrics for comparing the distributions, and will dump a large number of plots into the `plots/` folder.



## Todo:
1. requirements.txt
2. Clean up the scripts a little more.

