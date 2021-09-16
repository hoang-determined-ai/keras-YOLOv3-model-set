#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""
import os, time, random, argparse
import numpy as np
import tensorflow.keras.backend as K
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import LearningRateScheduler, TerminateOnNaN, LambdaCallback
# removed TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo5.model import get_yolo5_train_model
from yolo5.data import yolo5_data_generator_wrapper, Yolo5DataGenerator
from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper, Yolo3DataGenerator
from yolo2.model import get_yolo2_train_model
from yolo2.data import yolo2_data_generator_wrapper, Yolo2DataGenerator
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import EvalCallBack, CheckpointCleanCallBack, DatasetShuffleCallBack

### Import Determined packages
from determined.keras import TFKerasTrial, TFKerasTrialContext
from determined.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from typing import *

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)

class YoloV3Trial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        # Initialize the trial class.

        self.context = context

        # assign multiscale interval
        if self.context.get_hparam("multiscale"):
            self.rescale_interval = self.context.get_hparam("rescale_interval")
        else:
            self.rescale_interval = -1  # Doesn't rescale

        # model input shape check
        self.input_shape = (int(self.context.get_hparam("image_height")), int(self.context.get_hparam("image_width")))
        assert (self.input_shape[0] % 32 == 0 and self.input_shape[1] % 32 == 0), 'model_image_size should be multiples of 32'

        self.annotation_file = self.context.get_hparam("annotation_file")
        self.classes_path = self.context.get_hparam("classes_path")
        self.class_names = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = get_anchors(self.context.get_hparam("anchors_path"))
        self.num_anchors = len(self.anchors)

        if self.context.get_hparam("model_type").startswith('scaled_yolo4_') or self.context.get_hparam("model_type").startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, use yolo5 submodule but now still yolo3 data generator
            print("!@# Getting Yolo5")
            self.get_train_model = get_yolo5_train_model
        elif self.context.get_hparam("model_type").startswith('yolo3_') or self.context.get_hparam("model_type").startswith('yolo4_'):
            print("!@# Getting Yolo3")
            self.get_train_model = get_yolo3_train_model

    def build_model(self):
        # Define and compile model graph.

        # get freeze level according to CLI option
        if self.context.get_hparam("weights_path") != "None":
            freeze_level = 0
        else:
            freeze_level = 1

        #if self.context.get_hparam("freeze_level") is not None:
        #    freeze_level = self.context.get_hparam("freeze_level")

        # prepare optimizer
        optimizer = self.context.wrap_optimizer(get_optimizer(self.context.get_hparam("optimizer"),
                                                              self.context.get_hparam("learning_rate"),
                                                              average_type=None,
                                                              decay_type=None))

        # get normal train model
        model = self.context.wrap_model(
            self.get_train_model(self.context.get_hparam("model_type"),
                                 self.anchors,
                                 self.num_classes,
                                 weights_path=None,
                                 #weights_path=self.context.get_hparam("weights_path"),
                                 freeze_level=0,
                                 #freeze_level=freeze_level,
                                 optimizer=optimizer,
                                 label_smoothing=self.context.get_hparam("label_smoothing"),
                                 elim_grid_sense=self.context.get_hparam("elim_grid_sense"),
                                #model_pruning=self.context.get_hparam("model_pruning"),
                                #pruning_end_step=pruning_end_step
                                )
        )
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        return model

    def build_training_data_loader(self):
        # Create the training data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.

        # get train&val dataset
        dataset = get_dataset(self.annotation_file)
        if self.context.get_hparam("val_annotation_file") != "None":
            val_dataset = get_dataset(self.context.get_hparam("val_annotation_file"))
            num_train = len(dataset)
            num_val = len(val_dataset)
            dataset.extend(val_dataset)
        else:
            val_split = self.context.get_hparam("val_split")
            num_val = int(len(dataset) * val_split)
            num_train = len(dataset) - num_val

        if self.context.get_hparam("model_type").startswith('scaled_yolo4_') or self.context.get_hparam("model_type").startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, use yolo5 submodule but now still yolo3 data generator
            # TODO: create new yolo5 data generator to apply YOLOv5 anchor assignment
            # get_train_model = get_yolo5_train_model
            # data_generator = yolo5_data_generator_wrapper

            # tf.keras.Sequence style data generator
            self.train_data_generator = self.context.wrap_dataset(
                Yolo5DataGenerator(dataset[:num_train],
                                   self.context.get_per_slot_batch_size(),
                                   self.input_shape,
                                   self.anchors,
                                   self.num_classes,
                                   self.context.get_hparam("enhance_augment"),
                                   self.rescale_interval,
                                   self.context.get_hparam("multi_anchor_assign")
                                   )
            )
            self.val_data_generator = self.context.wrap_dataset(
                Yolo5DataGenerator(dataset[num_train:],
                                   self.context.get_per_slot_batch_size(),
                                   self.input_shape,
                                   self.anchors,
                                   self.num_classes,
                                   multi_anchor_assign=self.context.get_hparam("multi_anchor_assign")
                                   )
            )
            tiny_version = False

        elif self.context.get_hparam("model_type").startswith('yolo3_') or self.context.get_hparam("model_type").startswith('yolo4_'):
            # if num_anchors == 9:
            # YOLOv3 & v4 entrance, use 9 anchors
            # get_train_model = get_yolo3_train_model
            # data_generator = yolo3_data_generator_wrapper

            # tf.keras.Sequence style data generator
            self.train_data_generator = self.context.wrap_dataset(
                Yolo3DataGenerator(dataset[:num_train],
                                   self.context.get_per_slot_batch_size(),
                                   self.input_shape,
                                   self.anchors,
                                   self.num_classes,
                                   self.context.get_hparam("enhance_augment"),
                                   self.rescale_interval,
                                   self.context.get_hparam("multi_anchor_assign")
                                   )
            )
            self.val_data_generator = self.context.wrap_dataset(
                Yolo3DataGenerator(dataset[num_train:],
                                   self.context.get_per_slot_batch_size(),
                                   self.input_shape,
                                   self.anchors,
                                   self.num_classes,
                                   multi_anchor_assign=self.context.get_hparam("multi_anchor_assign")
                                   )
            )
            tiny_version = False

        return self.train_data_generator

    def build_validation_data_loader(self):
        # Create the validation data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.

        return self.val_data_generator

    def keras_callbacks(self) -> List[tf.keras.callbacks.Callback]:

        # Use Determined callbacks (note: no log_dir required for Tensorboard callback)
        logging = TensorBoard(histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0,
                                      min_lr=1e-10)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
        terminate_on_nan = TerminateOnNaN()



        callbacks = [logging, reduce_lr, early_stopping, terminate_on_nan]

        return callbacks

'''
def main_old(args):
    #log_dir = os.path.join('logs', '000')

    # callbacks for training process
    # checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
    #    monitor='val_loss',
    #    mode='min',
    #    verbose=1,
    #    save_weights_only=False,
    #    save_best_only=True,
    #    period=1)
    # checkpoint_clean = CheckpointCleanCallBack(log_dir, max_val_keep=5, max_eval_keep=2)

    # get different model type & train&val data generator

    elif self.context.get_hparam(model_type.startswith('tiny_yolo3_') or self.context.get_hparam(model_type.startswith('tiny_yolo4_'):
    #elif num_anchors == 6:
        # Tiny YOLOv3 & v4 entrance, use 6 anchors
        get_train_model = get_yolo3_train_model
        data_generator = yolo3_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo3DataGenerator(dataset[:num_train], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, self.context.get_hparam(enhance_augment, rescale_interval, self.context.get_hparam(multi_anchor_assign)
        #val_data_generator = Yolo3DataGenerator(dataset[num_train:], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, multi_anchor_assign=self.context.get_hparam(multi_anchor_assign)

        tiny_version = True
    elif self.context.get_hparam(model_type.startswith('yolo2_') or self.context.get_hparam(model_type.startswith('tiny_yolo2_'):
    #elif num_anchors == 5:
        # YOLOv2 & Tiny YOLOv2 use 5 anchors
        get_train_model = get_yolo2_train_model
        data_generator = yolo2_data_generator_wrapper

        # tf.keras.Sequence style data generator
        #train_data_generator = Yolo2DataGenerator(dataset[:num_train], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, self.context.get_hparam(enhance_augment, rescale_interval)
        #val_data_generator = Yolo2DataGenerator(dataset[num_train:], self.context.get_hparam(batch_size, input_shape, anchors, num_classes)

        tiny_version = False


    # prepare online evaluation callback
    if self.context.get_hparam(eval_online:
        eval_callback = EvalCallBack(self.context.get_hparam(model_type, dataset[num_train:], anchors, class_names, self.context.get_hparam(model_image_size, self.context.get_hparam(model_pruning, log_dir, eval_epoch_interval=self.context.get_hparam(eval_epoch_interval, save_eval_checkpoint=self.context.get_hparam(save_eval_checkpoint, elim_grid_sense=self.context.get_hparam(elim_grid_sense)
        callbacks.insert(-1, eval_callback) # add before checkpoint clean

    # prepare train/val data shuffle callback
    if self.context.get_hparam(data_shuffle):
        shuffle_callback = DatasetShuffleCallBack(dataset)
        callbacks.append(shuffle_callback)

    # prepare model pruning config
    pruning_end_step = np.ceil(1.0 * num_train / self.context.get_hparam(batch_size).astype(np.int32) * self.context.get_hparam(total_epoch)
    if self.context.get_hparam(model_pruning):
        pruning_callbacks = [sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)]
        callbacks = callbacks + pruning_callbacks

    # support multi-gpu training
    if self.context.get_hparam(gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(self.context.get_hparam(gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model
            model = get_train_model(self.context.get_hparam(model_type, anchors, num_classes, weights_path=self.context.get_hparam(weights_path, freeze_level=freeze_level, optimizer=optimizer, label_smoothing=self.context.get_hparam(label_smoothing, elim_grid_sense=self.context.get_hparam(elim_grid_sense, model_pruning=self.context.get_hparam(model_pruning, pruning_end_step=pruning_end_step)

    else:
        pass


    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = self.context.get_hparam(init_epoch
    epochs = initial_epoch + self.context.get_hparam(transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, self.context.get_hparam(batch_size, input_shape))
    #model.fit_generator(train_data_generator,
    model.fit_generator(data_generator(dataset[:num_train], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, self.context.get_hparam(enhance_augment, rescale_interval, multi_anchor_assign=self.context.get_hparam(multi_anchor_assign),
            steps_per_epoch=max(1, num_train//self.context.get_hparam(batch_size),
            #validation_data=val_data_generator,
            validation_data=data_generator(dataset[num_train:], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, multi_anchor_assign=self.context.get_hparam(multi_anchor_assign),
            validation_steps=max(1, num_val//self.context.get_hparam(batch_size),
            epochs=epochs,
            initial_epoch=initial_epoch,
            #verbose=1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)

    # Wait 2 seconds for next stage
    time.sleep(2)

    if self.context.get_hparam(decay_type or self.context.get_hparam(average_type:
        # rebuild optimizer to apply learning rate decay or weights averager,
        # only after unfreeze all layers
        if self.context.get_hparam(decay_type:
            callbacks.remove(reduce_lr)

        if self.context.get_hparam(average_type == 'ema' or self.context.get_hparam(average_type == 'swa':
            # weights averager need tensorflow-addons,
            # which request TF 2.x and have version compatibility
            import tensorflow_addons as tfa
            callbacks.remove(checkpoint)
            avg_checkpoint = tfa.callbacks.AverageModelCheckpoint(filepath=os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                                  update_weights=True,
                                                                  monitor='val_loss',
                                                                  mode='min',
                                                                  verbose=1,
                                                                  save_weights_only=False,
                                                                  save_best_only=True,
                                                                  period=1)
            callbacks.insert(-1, avg_checkpoint) # add before checkpoint clean

        steps_per_epoch = max(1, num_train//self.context.get_hparam(batch_size)
        decay_steps = steps_per_epoch * (self.context.get_hparam(total_epoch - self.context.get_hparam(init_epoch - self.context.get_hparam(transfer_epoch)
        optimizer = get_optimizer(self.context.get_hparam(optimizer, self.context.get_hparam(learning_rate, average_type=self.context.get_hparam(average_type, decay_type=self.context.get_hparam(decay_type, decay_steps=decay_steps)

    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")
    if self.context.get_hparam(gpu_num >= 2:
        with strategy.scope():
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, self.context.get_hparam(batch_size, input_shape))
    #model.fit_generator(train_data_generator,
    model.fit_generator(data_generator(dataset[:num_train], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, self.context.get_hparam(enhance_augment, rescale_interval, multi_anchor_assign=self.context.get_hparam(multi_anchor_assign),
        steps_per_epoch=max(1, num_train//self.context.get_hparam(batch_size),
        #validation_data=val_data_generator,
        validation_data=data_generator(dataset[num_train:], self.context.get_hparam(batch_size, input_shape, anchors, num_classes, multi_anchor_assign=self.context.get_hparam(multi_anchor_assign),
        validation_steps=max(1, num_val//self.context.get_hparam(batch_size),
        epochs=self.context.get_hparam(total_epoch,
        initial_epoch=epochs,
        #verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        callbacks=callbacks)

    # Finally store model
    if self.context.get_hparam(model_pruning:
        model = sparsity.strip_pruning(model)
    model.save(os.path.join(log_dir, 'trained_final.h5'))
'''