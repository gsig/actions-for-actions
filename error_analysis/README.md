# What Actions are Needed for Understanding Human Actions in Videos?
Diagnostic tools from "What Actions are Needed for Understanding Human Actions in Videos?" ICCV 2017

Contributor: Gunnar A. Sigurdsson

This tool analyses a Charades submission file (preferrably localization type) to generate Figure 3 in the paper.
That is, fraction of top ranked predictions for each class that are correct (TP), on the boundary (BND), other class with same object (OBJ), other class with same verb (VRB), other class with neither (OTH), or no class (FP).

# Requirements: 
python 2.7, numpy
```
(sudo) pip install numpy
```

# Usage:
python errorplot.py submission_file.txt

The script is meant to be edited and experimented with. There are constants at the top of the script that can be edited to create different figures, and it should be easy to add more attributes as needed.

Additional Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields

## Example 4:
Original settings. Plot for analyzing the different error made by Two-Stream. See example4.pdf.
```
sh get_test_submission_localize.sh
python errorplot.py test_submission_localize.txt
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

