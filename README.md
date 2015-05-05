Deeply-supervised Nets
=======
**Update**
For experiments on MNIST, cifar-100 and SVHN, please use the same architecture as CIFAR-10 provided. This architecture/hyper-parameter settings can generalize pretty well and can achieve even better results than the numbers reported in the paper. 

Also for Cifar-100 and SVHN, we used **softmax losses** instead of hinge losses for the output supervision and deep supervision. There are known convergence issues of hinge loss on > 10 classes in Caffe.

**Deeply Supervised Nets**
This DSN code is based on an older version of [CAFFE](http://caffe.berkeleyvision.org) framework. This is for reproducing the results reported in our [paper](http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/). Though with minimum engineering efforts you can apply this idea to your own code, as well as new network architectures.

We preprocess the data following the methods used in maxout networks and network in network paper. Please find the details [here](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout). Basically we only did GCN (global contrast normalization) on the benchmark datasets. Note that the scale of the data is [0,1] instead of [0, 255]. This is a tricky part when you use your own data: You should tune the learning rate accordingly. Also tools/cifar-float-data might be useful if you want to generate your own leveldb database from the gcn processed data. 

To reproduce the results more easily, you can also download our processed Leveldb files [here](https://drive.google.com/file/d/0B3j4tSL88j-eU0c4NGd4NC0tUVU/view?usp=sharing)

The configuration files are in the examples folder. You can run train\_full.sh script and it will automatically complete the training. The final result may vary from machine to machine. For cifar-10 you should be able to get at least 9.65% error, please contact me if you have problem in reproducing this. Thank you.

====
**Below is original caffe README, please follow the installation instructions to compile this code.**

Created by [Yangqing Jia](http://daggerfs.com), UC Berkeley EECS department.
In active development by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu/)).

## Introduction

Caffe aims to provide computer vision scientists with a **clean, modifiable
implementation** of state-of-the-art deep learning algorithms. Network structure
is easily specified in separate config files, with no mess of hard-coded
parameters in the code. Python and Matlab wrappers are provided.

At the same time, Caffe fits industry needs, with blazing fast C++/Cuda code for
GPU computation. Caffe is currently the fastest GPU CNN implementation publicly
available, and is able to process more than **20 million images per day** on a
single Tesla K20 machine \*.

Caffe also provides **seamless switching between CPU and GPU**, which allows one
to train models with fast GPUs and then deploy them on non-GPU clusters with one
line of code: `Caffe::set_mode(Caffe::CPU)`.

Even in CPU mode, computing predictions on an image takes only 20 ms when images
are processed in batch mode.

* [Installation instructions](http://caffe.berkeleyvision.org/installation.html)
* [Caffe presentation](https://docs.google.com/presentation/d/1lzyXMRQFlOYE2Jy0lCNaqltpcCIKuRzKJxQ7vCuPRc8/edit?usp=sharing) at the Berkeley Vision Group meeting

\* When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012.

## License

Caffe is BSD 2-Clause licensed (refer to the
[LICENSE](http://caffe.berkeleyvision.org/license.html) for details).

The pretrained models published by the BVLC, such as the
[Caffe reference ImageNet model](https://www.dropbox.com/s/n3jups0gr7uj0dv/caffe_reference_imagenet_model)
are licensed for academic research / non-commercial use only. However, Caffe is
a full toolkit for model training, so start brewing your own Caffe model today!

## Citing Caffe

Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }

## Documentation

Tutorials and general documentation are written in Markdown format in the `docs/` folder.
While the format is quite easy to read directly, you may prefer to view the whole thing as a website.
To do so, simply run `jekyll serve -s docs` and view the documentation website at `http://0.0.0.0:4000` (to get [jekyll](http://jekyllrb.com/), you must have ruby and do `gem install jekyll`).

We strive to provide provide lots of usage examples, and to document all code in docstrings.
We'd appreciate your contribution to this effort!

## Contributing

Caffe is developed with active participation of the community by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/).
We welcome all contributions!

Our workflow is this:

- The `dev` branch is for new development, community contributions, and testing.
- The `master` branch is handled by BVLC, which will integrate changes from `dev` on a roughly monthly schedule.
- Do new development in [feature branches](https://www.atlassian.com/git/workflows#!workflow-feature-branch) with decriptive names.
- Bring your work up-to-date by [rebasing](http://git-scm.com/book/en/Git-Branching-Rebasing) onto the latest `dev`. (Polish your changes by [interactive rebase](https://help.github.com/articles/interactive-rebase), if you'd like.)
- [Pull request](https://help.github.com/articles/using-pull-requests) your contribution to BVLC/caffe's `dev` branch for discussion and review.
  * PRs should live fast, die young, and leave a beautiful merge. Pull
    request sooner than later so that discussion can guide development.
  * Code must be accompanied by documentation and tests at all times.
  * Only fast-forward merges will be accepted.

See our [development guidelines](http://caffe.berkeleyvision.org/development.html) for further details–the more closely these are followed, the sooner your work will be merged.

#### [Shelhamer's](https://github.com/shelhamer) “life of a branch in four acts”

Make the `feature` branch off of the latest `bvlc/dev`
```
git checkout dev
git pull upstream dev
git checkout -b feature
# do your work, make commits
```

Prepare to merge by rebasing your branch on the latest `bvlc/dev`
```
# make sure dev is fresh
git checkout dev
git pull upstream dev
# rebase your branch on the tip of dev
git checkout feature
git rebase dev
```

Push your branch to pull request it into `dev`
```
git push origin feature
# ...make pull request to dev...
```

Now make a pull request! You can do this from the command line (`git pull-request -b dev`) if you install [hub](https://github.com/github/hub).

The pull request of `feature` into `dev` will be a clean merge. Applause.
>>>>>>> experiment-no-polish
