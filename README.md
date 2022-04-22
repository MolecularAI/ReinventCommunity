# ReinventCommunity (`jupyter` notebook tutorials for `REINVENT 3.2`)
This repository is a collection of useful `jupyter` notebooks, code snippets and example `JSON` files illustrating the use of [Reinvent 3.2](https://github.com/MolecularAI/Reinvent).
At the moment, the following notebooks are supported:
* `Complete_Use-Case_DRD2_Demo`: a full-fledged use case using public data on `DRD2`, including use of predictive models and elucidating general considerations
* `Create_Model_Demo`: explanation on how to initialize a new model (prior / agent) for `REINVENT` which can be trained in a *transfer learning* setup
* `Data_Preparation`: tutorial on how to prepare (clean, filter and standardize) data from a source such as `ChEMBL` to be used for training
* `Model_Building_Demo`: shows how to train a predictive (QSAR) model to be used with `REINVENT` based on the public `DRD2` dataset (classification problem)
* `Reinforcement_Learning_Demo`: example reinforcement learning run with a selection of scoring function components to generate novel compounds with ever higher scores iteratively
* `Reinforcement_Learning_Demo_Selectivity`: example illustrating the use of the relatively complicated `selectivity_component` to optimize potency against a target while simultaneously pushing for a low potency against one or more off-targets
* `Reinforcement_Learning_Demo_Tanimoto`: very simple (only 1, easy-to-understand component) transfer learning example
* `Reinforcement_Learning_Exploitation_Demo`: illustrates the exploitation scenario, where one is after solutions from a subspace in chemical space already well defined
* `Reinforcement_Learning_Exploration_Demo`: illustrates the exploration scenario, where the aim is to generate a varied set of solutions to a less stringently defined problem
* `Reinforcement_Learning_Demo_DockStream`: illustrates the use of `DockStream` in REINVENT, allowing the generative model to gradually optimize the docking score of proposed compounds. For more information on `DockStream`, please see the `DockStream` repository and the corresponding `DockStreamCommunity` repository for tutorial notebooks on `DockStream` as a standalone molecular docking tool.
* `Reinforcement_Learning_Demo_Icolos`: illustrates the use of [Icolos](https://github.com/MolecularAI/Icolos) in REINVENT using a docking scenario.
* `Sampling_Demo`: once an agent has been trained and is producing interesting results, it can be used to generate more compounds without actually changing it further - this is facilitated by the `sampling mode`
* `Score_Transformations`: as many components produce scores on an arbitrary scale, but `REINVENT` needs to receive it normalized to be a number between 0 and 1 (with values close to 1 meaning "good"), score transformations have been implemented and can be used as shown in this tutorial
* `Scoring_Demo`: in case a set of existing compound definitions (for example prior to starting a project) should be scored with a scoring function definition, the `scoring mode` can be used
* `Transfer_Learning_Demo`: this tutorial illustrates the `transfer learning` mode, which usually is used to "pre-train" an agent before `reinforcement learning` in case no adequate naive prior is available or to focus an already existing agent further
* `Transfer_Learning_Demo_Teachers_Forcing`: same as `Transfer_Learning_Demo` above, with explanation of `teachers forcing`
* `Lib-INVENT_RL1_QSAR`: Lib-INVENT example reinforcement learning run using a QSAR model
* `Lib-INVENT_RL2_QSAR_RF`: Lib-INVENT example reinforcement learning run using a random forest (RF) QSAR model
* `Lib-INVENT_RL3_ROCS_RF`: Lib-INVENT example reinforcement learning using OpenEye's ROCS 3D similarity (requires an OpenEye license)
* `Link-INVENT_RL`: Link-INVENT example reinforcement learning
* `Automated_Curriculum_Learning_demo`: illustrates the automated curriculum learning running model. The example demonstrates how to set-up a curriculum to guide the REINVENT agent to sample a target molecular scaffold. This scenario represents a complex objective as the target scaffold is not present in the training set for the prior model
