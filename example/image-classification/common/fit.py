# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import logging
import os
import time
from mxnet import profiler
from os.path import expanduser
import socket

import math
import warnings
import numpy as np

import mxnet as mx
from mxnet import kvstore
from mxnet import module
from mxnet import metric
from mxnet import ndarray
from mxnet.context import cpu
from mxnet.model import BatchEndParam,_update_params_on_kvstore
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
from mxnet import context as ctx
from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc
from threading import Thread

class MyModule(mx.mod.Module):
    
    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
        ####chris_arg
        if int(os.getenv("TASK_LIMIT", 0)) != 0:  #为0时不分task限制，为1时分task但是每轮更新，为2时分task并但固定
            get_task_cmd = "sh /home/ubuntu/tc.sh -l 1"
        else:
            self.logger.info("no_task_bandwidth_limit")
            get_task_cmd = "sh /home/ubuntu/tc.sh -l 0"
        os.system(get_task_cmd)
        delay_time = float(os.getenv("DELAY_TIME",0.8))
        ps_upload_bandwidth_part1 = int(os.getenv("PS_UPLOAD_BANDWIDTH1",2000))
        worker_upload_bandwidth_part1 = int(os.getenv("WORKER_UPLOAD_BANDWIDTH1",2000))
        ps_upload_bandwidth_part2 = int(os.getenv("PS_UPLOAD_BANDWIDTH2",2000))
        worker_upload_bandwidth_part2 = int(os.getenv("WORKER_UPLOAD_BANDWIDTH2",2000))
        tc_command = "sudo tc class change dev {} parent 1: classid 1:3 htb rate {}mbit ceil {}mbit  && sudo tc class change dev {} parent 1: classid 1:4 htb rate {}mbit ceil {}mbit" 
        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward(data_batch, is_train=True)
                if int(os.getenv("TASK_LIMIT", 0)) == 1:
                    ##first part bandwidth allocation
                    ndarray.waitall()
                    # self.logger.info("change bandwidth part1:, "+str(time.time()))
                    x = str(ps_upload_bandwidth_part1)
                    y = str(worker_upload_bandwidth_part1)
                    cmd_up = tc_command.format("ens3", x, x, "ens3", y, y)
                    cmd_down = tc_command.format("ifb0", y, y, "ifb0", x, x)
                    os.system(cmd_up)
                    os.system(cmd_down)
                # self.logger.info("after forward, "+str(time.time()))
                self.backward()
                # self.logger.info("before update: "+str(time.time()))
                self.update() #异步执行的
                if int(os.getenv("TASK_LIMIT", 0)) == 1:
                    x = str(ps_upload_bandwidth_part2)
                    y = str(worker_upload_bandwidth_part2)
                    cmd_up = tc_command.format("ens3", x, x, "ens3", y, y)
                    cmd_down = tc_command.format("ifb0", y, y, "ifb0", x, x)
                    time.sleep(delay_time) 
                    ##second part bandwidth allocation
                    # self.logger.info("change bandwidth part2:, "+str(time.time()))
                    os.system(cmd_up)
                    os.system(cmd_down)
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True

                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()



def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    train.add_argument('--gc-type', type=str, default='none',
                       help='type of gradient compression to use, \
                             takes `2bit` or `none` for now')
    train.add_argument('--gc-threshold', type=float, default=0.5,
                       help='threshold for 2bit gradient compression')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args, kv)
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()

        return


    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.MyModule(
        context       = devs,
        symbol        = network
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
        'learning_rate': lr,
        'wd' : args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    logfile = expanduser("~")+"/profiler-"+str(socket.gethostname())+"-"+str(kv.rank)+".json"
    mx.profiler.profiler_set_config(mode='all', filename=logfile)
    def callback():
        def switch_profiler(param):
            if param.epoch == 0 and param.nbatch == 100:
                profiler.profiler_set_state('run')
            if param.epoch == 0 and param.nbatch == 110:
                profiler.profiler_set_state('stop')
                profiler.dump_profile()

        return switch_profiler;

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches),
            callback()]

    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
              begin_epoch        = args.load_epoch if args.load_epoch else 0,
              num_epoch          = args.num_epochs,
              eval_data          = val,
              eval_metric        = eval_metrics,
              kvstore            = kv,
              optimizer          = args.optimizer,
              optimizer_params   = optimizer_params,
              initializer        = initializer,
              arg_params         = arg_params,
              aux_params         = aux_params,
              batch_end_callback = batch_end_callbacks,
              epoch_end_callback = checkpoint,
              allow_missing      = True,
              monitor            = monitor)
