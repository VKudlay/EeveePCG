# **Starter Project:** Generating Eeveelution Point Clouds With Point Set Generation Network
### Repo By Vadim Kudlay 

![eeveelution pics](objs/eevee_pic.png)

A full tour of the results and thought process behind the implementation can be found in [`Eeveelutions.ipynb`](Eeveelutions.ipynb) and its associated [html export](https://vkudlay.github.io/EeveePCG/).

This is a brief implementation of a point set generation network to attempt to generate different eeveelutions (i.e. the Pokémon evolutions). Specifically, the network is trained to map from an origin point cloud to a destination point cloud based on a 'transformation vector' which specifies which class the cloud should progress towards. The implementation largely utilizes an basic autoencoder architecture quite similar to [the one mentioned in the main reference paper](https://arxiv.org/abs/1707.02392).

---

## Requirements

If you would like to replicate the experiments without modifying the code much, the experiment and associated visualizations were implemented with the following libraries: 

- **Tensorflow** 2.5
- **Python** (~3.9.6)
- [**PyMesh**](https://pymesh.readthedocs.io/en/latest/installation.html#download-the-source) (~0.3)
- **Matplotlib** (~3.4.2)
- **Scipy** (~1.7.0)
- **Numpy** (~1.19.5)
- **Jupyterlab** (~3.0.16)

---

## Major References

- [Learning Representations and Generative Models for 3D Point Clouds (Achlioptas et. al. 2018)](https://arxiv.org/abs/1707.02392)
- [Generative Deep Learning with TensorFlow (Coursera; DeepLearning.AI)](https://www.coursera.org/learn/generative-deep-learning-with-tensorflow)
- [Build Basic Generative Adversarial Networks (Coursera; DeepLearning.AI)](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans)
- Additional implementation-level references documented alongside code. 

---

## Note 

A subset of the utility functions (specifically the [loss function implementations](models/losses.py)) may have been depricated if unused in the experiment; as such, they may require minor tuning and checking over if used in further analysis. These functions were used at some stage of the exploration and may have use in certain contexts but were not significant enough to include in the final analysis. 