# What Actions are Needed for Understanding Human Actions in Videos?
Diagnostic tools from "What Actions are Needed for Understanding Human Actions in Videos?" ICCV 2017

Contributor: Gunnar A. Sigurdsson

This tool can combine Charades submission files with the oracles to generate plots similar to Figure 9 in the paper.

# Requirements: 
python 2.7, numpy, pandas
```
(sudo) pip install numpy pandas
```

# Usage:
python oraclesplot.py submission_file1.txt submission_file2.txt etc...

The script is meant to be edited and experimented with. There are constants at the top of the script that can be edited to create different figures, and it should be easy to add more attributes as needed.

Additional Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields

## Example 3:
Original settings. Plot for comparing Two-Stream and TFields when combined with oracles. See example3.pdf.
```
python oraclesplot.py ../tool/Two-Stream.txt ../tool/TFields.txt
```


# BibTex
If this tool helps your research:
```
@inproceedings{sigurdsson2017actions,
author = {Gunnar A. Sigurdsson and Olga Russakovsky and Abhinav Gupta},
title = {What Actions are Needed for Understanding Human Actions in Videos?},
booktitle={International Conference on Computer Vision (ICCV)},
year={2017},
pdf = {http://arxiv.org/pdf/1708.02696.pdf},
code = {https://github.com/gsig/actions-for-actions},
}
```

