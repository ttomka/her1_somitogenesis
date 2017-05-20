# her1_somitogenesis
Author: Tomas Tomka
Part of my Master's thesis at ETH Zurich, D-BSSE
Supervised by Dr. Marcelo Boareto and Prof. Dr. Dagmar Iber

The code is structured in two files:
- the ipython notebook 2017_Somitogenesis_her1.ipynb contains mostly plotting routines required to get the final figures, it imports her1_dde_aux.py
- her1_dde_aux.py contains more elaborate methods, which simulate the differential delay equations (DDE) under various conditions and analyse/plot the output

The model is largely adopted from: Lewis, J. (2003). Autoinhibition with transcriptional delay: a simple mechanism for the zebrafish somitogenesis oscillator. Curr. Biol. 13, 1398-1408.

The travelling wave simulation setup is largely adopted from: Ay, A., Holland, J., Sperlea, A., Devakanmalai, G. S., Knierer, S., Sangervasi, S., Stevenson, A. and Özbudak,
E. M. (2014). Spatial gradients of protein-level time delays set the pace of the traveling segmentation clock
waves. Development 141, 4158-4167.

Background: Cells in the presomitic mesoderm of the developing embryo 'do the wave' with her genes (model organism: zebrafish), like people in a stadium who raise their hands. The emerging waves control the rhythmicity of somite formation (sequential epithelization of mesoderm cell patches, anterior-to-posterior). How is the gene activity in presomitic cells coordinated, such that travelling waves emerge?

Somitogenesis: We have designed a parsimonious model of the her1 oscillator, which is much simpler than recent models. It recapitulates the main principles at work:     
 - oscillation driven by autorepression
 - oscillations are synchronized locally
 - an increasing gradient in the period of oscillation
 
contact: tomas.tomka11@gmail.com
