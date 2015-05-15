Deeply-supervised Nets
=======
**Update**
For experiments on cifar-100 and SVHN, please use the same architecture as CIFAR-10 provided. This architecture/hyper-parameter settings can generalize pretty well and can achieve the numbers reported in the paper. 

Also for Cifar-100 and SVHN, we used **softmax losses** instead of hinge losses for the output supervision and deep supervision. There are known convergence issues of hinge loss on > 10 classes in Caffe.

Please cite DSN  in your publications if it helps your research:

   [1] Deeply-Supervised Nets Chen-Yu Lee*, Saining Xie*, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu (* indicates equal contributions) In Proceedings of AISTATS 2015

If you have problems reproducing the experiments feel free to contact the authors.

**Deeply Supervised Nets**
This DSN code is based on an older version of [CAFFE](http://caffe.berkeleyvision.org) framework. This is for reproducing the results reported in our [paper](http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/). Though with minimum engineering efforts you can apply this idea to your own code, as well as new network architectures.

We preprocess the data following the methods used in maxout networks and network in network paper. Please find the details [here](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout). Basically we only did GCN (global contrast normalization) on the benchmark datasets. Note that the scale of the data is [0,1] instead of [0, 255]. This is a tricky part when you use your own data: You should tune the learning rate accordingly. Also tools/cifar-float-data might be useful if you want to generate your own leveldb database from the gcn processed data. 

To reproduce the results more easily, you can also download our processed Leveldb files [here](https://drive.google.com/file/d/0B3j4tSL88j-eU0c4NGd4NC0tUVU/view?usp=sharing)

The configuration files are in the examples folder. You can run train\_full.sh script and it will automatically complete the training. The final result may vary from machine to machine. For cifar-10 you should be able to get at least 9.65% error, please contact me if you have problem in reproducing this. Thank you.
