#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin cifar-10_solver.prototxt 2>&1  | tee dsn.log
#GLOG_logtostderr=1 $TOOLS/train_net.bin finetune_cifar-10_solver.prototxt finetune_nosvm_cifar-10_iter_50000.solverstate 2>&1 | tee -a finetune_from_50000_without_svm.log
GLOG_logtostderr=1 $TOOLS/finetune_net.bin finetune_cifar-10_solver.prototxt cifar-10_iter_50000 2>&1 | tee finetune_from_50000_without_svm.log
GLOG_logtostderr=1 $TOOLS/finetune_net.bin finetune_cifar-10_solver_lr2.prototxt finetune_nosvm_cifar-10_iter_120000 2>&1 | tee finetune_lr2_from_120000_without_svm.log
#You should reach 90.3% accuracy after this
