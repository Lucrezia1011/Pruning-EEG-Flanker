# Pruning-EEG-Flanker

Analysis code from "Changes in Behavior and Neural Dynamics across Adolescent Development" DOI: https://doi.org/10.1523/JNEUROSCI.0462-23.2023 by Lucrezia Liuzzi, Daniel S. Pine,  Nathan A. Fox and Bruno B. Averbeck. 

Code is dependedent on functions from EEGLAB and FieldTrip toolbox: 

https://sccn.ucsd.edu/eeglab/index.php
EEGLAB: A Delorme & S Makeig (2004) EEGLAB: an open source toolbox for
analysis of single-trial EEG dynamics. Journal of Neuroscience Methods, 134, 9?21.

https://www.fieldtriptoolbox.org/
FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data. Robert Oostenveld, Pascal Fries, Eric Maris, and Jan-Mathijs Schoffelen. Computational Intelligence and Neuroscience, 2011; 2011:156869.


Data preprocessing was run through the UMADE Pipeline:
https://github.com/ChildDevLab/MADE-EEG-preprocessing-pipeline
************************************************************************
The Maryland Analysis of Developmental EEG (UMADE) Pipeline
Version 1.0
Developed at the Child Development Lab, University of Maryland, College Park

Contributors to MADE pipeline:
Ranjan Debnath (rdebnath@umd.edu)
George A. Buzzell (gbuzzell@umd.edu)
Santiago Morales Pamplona (moraless@umd.edu)
Stephanie Leach (sleach12@umd.edu)
Maureen Elizabeth Bowers (mbowers1@umd.edu)
Nathan A. Fox (fox@umd.edu)

MADE uses EEGLAB toolbox and some of its plugins. Before running the pipeline, you have to install the following:
EEGLab:  https://sccn.ucsd.edu/eeglab/downloadtoolbox.php/download.php

You also need to download the following plugins/extensions from here: https://sccn.ucsd.edu/wiki/EEGLAB_Extensions

Specifically, download:
MFFMatlabIO: https://github.com/arnodelorme/mffmatlabio/blob/master/README.txt
FASTER: https://sourceforge.net/projects/faster/
ADJUST: https://www.nitrc.org/projects/adjust/
Adjusted ADJUST (included in this pipeline):  https://github.com/ChildDevLab/MADE-EEG-preprocessing-pipeline

After downloading these plugins (as zip files), you need to place it in the eeglab/plugins folder.
For instance, for FASTER, you uncompress the downloaded extension file (e.g., 'FASTER.zip') and place it in the main EEGLAB "plugins" sub-directory/sub-folder.
After placing all the required plugins, add the EEGLAB folder to your path by using the following code:

addpath(genpath(('...')) % Enter the path of the EEGLAB folder in this line

Please cite the following references for in any manuscripts produced utilizing MADE pipeline:

EEGLAB: A Delorme & S Makeig (2004) EEGLAB: an open source toolbox for
analysis of single-trial EEG dynamics. Journal of Neuroscience Methods, 134, 9?21.

firfilt (filter plugin): developed by Andreas Widmann (https://home.uni-leipzig.de/biocog/content/de/mitarbeiter/widmann/eeglab-plugins/)

FASTER: Nolan, H., Whelan, R., Reilly, R.B., 2010. FASTER: Fully Automated Statistical
Thresholding for EEG artifact Rejection. Journal of Neuroscience Methods, 192, 152?162.

ADJUST: Mognon, A., Jovicich, J., Bruzzone, L., Buiatti, M., 2011. ADJUST: An automatic EEG
artifact detector based on the joint use of spatial and temporal features. Psychophysiology, 48, 229?240.
Our group has modified ADJUST plugin to improve selection of ICA components containing artifacts

This pipeline is released under the GNU General Public License version 3.
