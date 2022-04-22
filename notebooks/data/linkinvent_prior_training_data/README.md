# This folder contains all the necessary files to train the Link-INVENT prior from scratch.
# A trained prior is also provided in /models/linkinvent.prior

# There are 4 relevant files:
1) Training_Partition(1,2,3,4) - these are the SMILES used for training. They are partitioned into separate files for uploading reasons. One should recombine them to obtain the full training data.
2) Validation.smi - this file contains the validation SMILES
3) held_out_scaffolds.smi - this file contains the 287 unique Bemis-Murcko scaffolds that are in the Validation.smi file
4) reaction.smirks - this file contains all reaction SMIRKS used to slice the raw ChEMBL data to generate the training and validation sets