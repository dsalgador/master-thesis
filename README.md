# Solving a Complex Optimization Problem for Product Delivery with Reinforcement Learning and Artificial Neural Networks
Master's thesis about Reinforcement Learning (RL), Deep RL and Optimization.

* **Author**: Daniel Salgado Rojo
* **Tuto**r: Toni Lozano Bagén
* **University**: Autonomous University of Barcelona
* **Master**: Master's Degree in Modelling for Science and Engineering
* **Specialization**: Data Science

## Abstract

This master thesis is motivated by a real world problem for Product Delivery (PD) - an optimization
problem which combines inventory control and vehicle routing - inspired by a project from the
company, Grupo AIA, where I have been doing an internship from March to June 2018. The
solution proposed to the client uses classical constraint optimization techniques which tend to be
slow when finding optimal solutions and do not scale properly when the number of shops and trucks
used to deliver a product increases.

Machine Learning (ML) has become a very popular field in Data Science due to the increase in
computation power in recent years. It is usually said that ML techniques divide in two types,
Supervised Learning and Unsupervised Learning. However, this is not a complete classification.
Apart from these two types, we must distinguish another one which is very different from those
two: Reinforcement Learning (RL). RL consists of techniques that have been used for decades in
the field of Artificial Intelligence for many applications in fields such as robotics and industrial
automation [1], health and medicine [2, 3], Media and Advertising [4, 5, 6], Finance [7], text, speech
and dialog systems [8, 9], and so forth.

RL provides a nice framework to model a large variety of stochastic optimization problems [10].
Nevertheless, classical approaches to large RL problems suffer from three curses of dimensionality:
explosions in state and action spaces, and a large number of possible next states of an action
due to stochasticity [11, 12]. There is very few literature about the application of Reinforcement
Learning to a PD optimization framework. The only paper we have found that focus on the
practical application of RL to the domain of PD is from S. Proper and P. Tadepalli (2006) [12].
In that paper they propose a variant of a classical RL technique called ASH-learning, where they
use “tabular linear functions” (TLF) to learn the so-called H-values, which are then used to decide
how to control the product delivery system of interest. Proper and Tadepalli show the results of
controlling a simplistic and discretised system of 5 shops and 4 trucks with ASH-learning, where
the three curses of dimensionality are present, and the results are successful for that particular
example with an small number of trucks and shops. However, in practical situations, the number
of shops and trucks may be so large (for instance, lets say 30 shops and about 7 to 10 trucks) that
the explosion of the dimensionality of the state and action spaces would make those classical RL
techniques impractical.

In this thesis [13] we present a novel approach for solving product delivery problems by means
of Reinforcement Learning and Deep Neural Networks (DNN), a field also referred to as Deep
Reinforcement Learning (DRL). The idea is that the nonlinearity and complexity of DNN should
be better for learning to solve complex optimization problems than TLF, and the tabular functions
in general that have been used so far in classical RL. Moreover, we expect that DNN could be the
key to solve some of the curses of dimensionality such as the explosion of the state-action spaces;
in the framework of PD, we expect them to scale better than classical approaches to systems with
a large number of shops and trucks. In addition, we have developed an OpenAI gym environment
for our PD problem which is available in a GitHub repository [here](https://github.com/dsalgador/gym-pdsystem).

## Contents
The following subsections are the main parts of the work. The first one is about classical reinforcement learning (the Q-learning algorithm). The second one focus on classical supervised Machine Learning using Neural Networks in the context of multilabel calssification. The third part focus ona more recent reinforcement learning algorithm called Policy Gradient, and which by the use of Neural networks scales much better than Q-learning with the classical approach.

### 1. Q-learning (Classical Reinforcement Learning)
In [this](https://github.com/dsalgador/master-thesis/tree/master/Q-learning) first part (from Chapter 4) we introduce Q-learning, one of the most popular value-based algorithms aimed to learn the optimal Q-values and from them define an optimal policy. Although Q-learning algorithm is a bit antiquate, it will serve us as an starting point to learn classical reinforcement learning by applying it to our product delivery problem.

### 2. Imitation-Learning (Supervised Machine Learning: Classification)

In Chapter 5 we introduce the basic concepts about Artificial Neural Networks, more concretely
Deep Neural Networks (DNN). We start with an introduction of what is a NN model, and then focus
on how to train it. Finally we present an application of DNN to classification, for a toy example
related to the product delivery problem we are working with in this thesis. In [this](https://github.com/dsalgador/master-thesis/tree/master/Imitation-Learning) folder we can find the
notebooks and simulation folders for that part.

### 3. Policy-Gradient (Deep Reinforcement Learning)

In chapter 6 we aruse DNN to play the role of a parametrized policy $\pi_\theta$, and introduce
a particular type of algorithms, Policy Gradient (PG), that allow us to train the network to improve
the policy by means of simulated episodes. The field where there are used Deep Neural Networks
to solve Reinforcement Learning problems is called Deep Reinforcement Learning (DRL). In [this](https://github.com/dsalgador/master-thesis/tree/master/Policy-Gradient) folder we have put all simulations and notebooks related to that part.


## References
* [1] Jens Kober, J. Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey.
Int. J. Rob. Res., 32(11):1238–1274, September 2013.
* [2] Michael J. Frank, Lauren C. Seeberger, and Randall C. O’Reilly. By carrot or by stick: Cog-
nitive reinforcement learning in parkinsonism. Science, 306(5703):1940–1943, 2004.
* [3] Zhao Yufan, Kosorok Michael R., and Zeng Donglin. Reinforcement learning design for cancer
clinical trials. Statistics in Medicine, 28(26):3294–3315.
* [4] Alekh Agarwal, Sarah Bird, Markus Cozowicz, Luong Hoang, John Langford, Stephen Lee,
Jiaji Li, Dan Melamed, Gal Oshri, Oswaldo Ribas, Siddhartha Sen, and Alex Slivkins. A
multiworld testing decision service. CoRR, abs/1606.03966, 2016.
* [5] Han Cai, Kan Ren, Weinan Zhang, Kleanthis Malialis, Jun Wang, Yong Yu, and Defeng Guo.
Real-time bidding by reinforcement learning in display advertising. CoRR, abs/1701.02490, 2017.
* [6] Naoki Abe, Naval Verma, Chid Apte, and Robert Schroko. Cross channel optimized marketing
by reinforcement learning. In Proceedings of the Tenth ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining, KDD ’04, pages 767–772, New York, NY, USA, 2004. ACM.
* [7] Francesco Bertoluzzo and Marco Corazza. Reinforcement learning for automated financial
trading: Basics and applications. In Simone Bassis, Anna Esposito, and Francesco Carlo
Morabito, editors, Recent Advances of Neural Network Models and Applications, pages 197–
213, Cham, 2014. Springer International Publishing.
* [8] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive
summarization. CoRR, abs/1705.04304, 2017.
* [9] Bhuwan Dhingra, Lihong Li, Xiujun Li, Jianfeng Gao, Yun-Nung Chen, Faisal Ahmed, and
Li Deng. Towards end-to-end reinforcement learning of dialogue agents for information access.
In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics.
ACL – Association for Computational Linguistics, July 2017.
* [10] Richard S. Sutton and Andrew G. Barto. Introduction to Reinforcement Learning. MIT Press,
Cambridge, MA, USA, 1st edition, 1998.
* [11] W. B. Powell, A. George, B. Bouzaiene-Ayari, and H. P. Simao. Approximate dynamic pro-
gramming for high dimensional resource allocation problems. In Proceedings. 2005 IEEE In-
ternational Joint Conference on Neural Networks, 2005., volume 5, pages 2989–2994 vol. 5,
July 2005.

* [12] Scott Proper and Prasad Tadepalli. Scaling model-based average-reward reinforcement learning
for product delivery. In Johannes Fürnkranz, Tobias Scheffer, and Myra Spiliopoulou, editors,
Machine Learning: ECML 2006, pages 735–742, Berlin, Heidelberg, 2006. Springer Berlin
Heidelberg.

For the rest of the references see the full document [here](https://github.com/dsalgador/master-thesis/blob/master/thesis.pdf)

## Prerequisites
* Python (tested for python 3.5 and 3.6)
* The gym-pdsystem python package is needed due to some of the python libraries that are found there. 
  Just clone the repository from [here](https://github.com/dsalgador/gym-pdsystem/tree/master/gym_pdsystem)
* Tensorflow library

## License
This work is under a GNU [license](https://github.com/dsalgador/master-thesis/blob/master/LICENSE)
