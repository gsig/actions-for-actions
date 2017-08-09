# What Actions are Needed for Understanding Human Actions in Videos?
Diagnostic tools and additional visualizations from "What Actions are Needed for Understanding Human Actions in Videos?" ICCV 2017

Contributor: Gunnar A. Sigurdsson

Will be updated closer to ICCV 2017

# Diagnostic Tool for Charades

To develop better algorithms for understanding activities, it will help to understand how and why they work. Looking solely at a single performance number obscures away from various details about the algorithms. To facilitate this process we are releasing a set of open-source Python scripts to generate detailed visualizations of one or more algorithms. The tool operates on the same file format required for the official Charades evaluation scripts, and therefore only needs a single submission file from each algorithm for analysis and comparison.
The tool contains annotations for many attributes, and any subset of those can be selected for visualization. This allows to quickly generate diagnostic plots of one or more algorithms that fit into $1/4$ of a page. These plots follow the same structure as the experiments section of the paper. A list of attributes and abbreviations is presented at the end of this section.
We note that this analysis could be extended to any dataset, by following our methodology outlined in the paper to compute and collect the same attributes.
To highlight the usefulness of the tool we consider use cases below.

# Single Algorithm Diagnostics for Classification and Localization

First we visualize how individual baselines would look under the diagnostic. Output from the diagnostics tool is presented for multiple baselines on Charades. The same can be done for any algorithm that can do both classification and localization.

Diagnostic Output on Charades includes classification and localization. *0,1,2,3\}* denotes the degree of each continuous attribute and *N*,*Y* denotes if a binary attributes is present (Yes) or absent (No). Dashed lines indicate the overall performance of each algorithm. Blue line is classification and Brown line is localization performance.

*If you want your algorith to appear here, you are welcome to send me an email with links to your classification and/or localization submission files on the Charades test set!*

## Two-Stream
![Two-Stream](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplot_twostream.png.jpg)

## IDT
![Two-Stream](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplot_idt.png.jpg)

## LSTM
![Two-Stream](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplot_lstm.png.jpg)

## ActionVLAD
![Two-Stream](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplot_actionvlad.png.jpg)

## Asynchromous Temporal Fields
![Two-Stream](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplot_tfields.png.jpg)

# Multiple Algorithm Comparison

To understand the difference between multiple algorithms, or different versions of the same algorithm, the tool can also be used to visualize many different algorithms. A condensed summary of the plots in the paper are presented in below for classification on Charades.

![Algorithm comparison](https://dl.dropboxusercontent.com/u/10728218/web/output_megaplotall.png.jpg)


# List of Attributes and Their Abbreviations
Below we list the abbreviations used in the plots and their meanings. The continuous attributes are grouped into varying degrees of the attribute where *1* is the attribute is low and *4* the attribute is high.

* **\#Samples**: The number of occurrences of an action across the training videos.
* **\#Frames**: The number of frames where the activity is taking place.
* **Novel**: Smallest edit distance between the sequence of activities in a test video and any video in the training set, and normalize by subtracting the average distance for all videos with that number of activities.
* **NewActor**: The subject/environment in the video was not seen in the training set.
* **Extent**: The average temporal extent of activities in each category in the training set.
* **Extent-v**: The average temporal extent of activities in each video.
* **Seq**: Is the category commonly part of a sequence or does it happen in isolation, such as "holding a cup" compared to "running".
* **Short**: Is the activity a brief activity such as "putting down cup".
* **Passive**: Does the activity happen in the background such as "sitting on a couch".
* **AvgMotion**: Average motion in each instance of the category on average.
* **AvgMotion-v**: Average motion in a given video by looking at the amount of optical flow in the video.
* **PersonSize**: The average size of a person in a video as measured by the highest-scoring per-frame Faster-RCNN bounding box detection.
* **People**: Whether there are more than one person in the video.
* **PoseVar**: The average Procrustes distance between any two poses in the category.
* **PoseRatio**: The ratio between the average pose variability within an instance and the pose variability for the category in general.
* **Obj**: If the category involves an object such as “drinking from a cup” compared to "running"
* **\#Obj**: How many categories share an object with the given category
* **\#Verbs**: How many categories share an verb with the given category
* **Interact**: If the activity involves physically interacting with the environment.
* **Tool**: The activity involves the use of a tool for interacting with the environment.
* **\#Actions**: The number of activities that occur in a video.
