# What Actions are Needed for Understanding Human Actions in Videos?
Diagnostic tools from "What Actions are Needed for Understanding Human Actions in Videos?" ICCV 2017

Contributor: Gunnar A. Sigurdsson

To develop better algorithms for understanding activities, it will help to understand how and why they work. Looking solely at a single performance number obscures away from various details about the algorithms. To facilitate this process we are releasing a set of open-source Python scripts to generate detailed visualizations of one or more algorithms. The tool operates on the same file format required for the official Charades evaluation scripts, and therefore only needs a single submission file from each algorithm for analysis and comparison.
The tool contains annotations for many attributes, and any subset of those can be selected for visualization. This allows to quickly generate diagnostic plots of one or more algorithms that fit into 1/4 of a page. These plots follow the same structure as the experiments section of the paper. 

# Requirements: 
python 2.7, numpy, pandas
```
(sudo) pip install numpy pandas
```

# Usage:
python charadesplot.py submission_file1.txt submission_file2.txt etc..

The script is meant to be edited and experimented with. There are constants at the top of the script that can be edited to create different figures, and it should be easy to add more attributes as needed.

Additional Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields

## Example 1:
Original settings. Analysis plot for a single algorithm with error bars. See example1.pdf.
```
python charadesplot.py Two-Stream.txt
```

## Example 2:
Original settings except:
```
YLIM = [0.05,0.40]
ERRORBARS = ...[0]
```
Comparison plot for multiple algorithms without error bars. See example2.pdf.
```
python charadesplot.py Two-Stream.txt TFields.txt
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

