from utils import *
import configs
import os, sys, time, tqdm, datetime

from keras.optimizers import SGD, Adam
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback
from vit_model import *


def main():
    # wandb.login()

    set_seed(configs.seed)

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')
    # log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    is_gpu = tf.config.list_physical_devices('GPU') is not None
    wandb_dir = '/root/autodl-tmp/dl_project12/wandb_logs' if is_gpu else \
        'D:/UPC_Course3/DL/dl_project1/wandb_logs'

    # initialize wandb logging to your project
    wandb.init(
        job_type='Training',
        project=configs.PROJECT_NAME,
        dir=wandb_dir,
        entity=configs.TEAM_NAME,
        config=configs.wandb_config,
        sync_tensorboard=True,
        name='cnn6' + now,
        notes='min_lr=0.00001',
        ####
    )

    config = wandb.config
    batch_size = config.batch_size
    num_classes = len(configs.data_classes)
    first_stage_epochs = config.first_stage_epochs
    finetune_epoches = config.finetune_epochs

    # freeze_blocks = 11
    lr = config.learning_rate
    epsilon = config.epsilon
    weight_decay = config.weight_decay
    amsgrad = config.amsgrad
    early_stopping = config.early_stopping
    activation = config.activation
    augment = config.augment

    print('Build Training dataset')
    X_train = tf.keras.utils.image_dataset_from_directory(
        configs.train_folder,
        batch_size=batch_size,  # batch_size
        # image_size=(img_height, img_width), # resize
        shuffle=True,
        seed=configs.seed
    )

    print('Build Validation dataset')
    X_val = tf.keras.utils.image_dataset_from_directory(
        configs.val_folder,
        batch_size=batch_size,  # batch_size
        # image_size=(img_height, img_width), # resize
        shuffle=False,
        seed=configs.seed,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    X_train = X_train.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    model = vit_base_patch16_224_in21k(num_classes=num_classes, has_logits=False)
    model.build((1, 224, 224, 3))

    # load weights
    pre_weights_path = './pretrain_weights/ViT-B_16.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    if config.optimizer == 'adam' and weight_decay == 0:
        optimizer = Adam(
            lr=lr,
            epsilon=epsilon,
            amsgrad=amsgrad
        )

    elif config.optimizer == 'adam' and weight_decay != 0:
        optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=lr,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

    # SGD + Momentum: Great if LR decayed properly.
    elif config.optimizer == "sgd" and config.weight_decay == 0:
        optimizer = SGD(
            learning_rate=lr,
            momentum=config.momentum,
            nesterov=config.nesterov
        )

    else:
        optimizer = tfa.optimizers.SGDW( weight_decay=config.weight_decay)

    early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                                   restore_best_weights=True)
    wandb_callback = WandbCallback(
                                    # save_model=False,
                                    # log_weights=True,
                                    # log_gradients=True,
                                  )
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.00001)

    for layer in model.layers:
        # if "pre_logits" not in layer.name and "head" not in layer.name:
        if layer.name in configs.unfreeze_layer_names:
            layer.trainable = False
        else:
            print("training {}".format(layer.name))

    # Compile after unfreeze classifier
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer, metrics=["accuracy"])

    print('Start first-stage training! Training the classifier of model!')
    t0 = time.time()
    history = model.fit(X_train,
                        validation_data=X_val,
                        epochs=first_stage_epochs,
                        # callbacks=[reduce_lr],
                        # callbacks=[early_callback, wandb_callback],
                        callbacks=[reduce_lr, wandb_callback],  # other callback?
                        )

    # Finetune Stage: unfreeze the top blocks to do finetune

    for layer in model.layers:
        if layer.name in configs.unfreeze_layer_names:
            layer.trainable = False
        else:
            print("training {}".format(layer.name))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(X_train,
                        validation_data=X_val,
                        epochs=finetune_epoches,
                        # callbacks=[reduce_lr],
                        # callbacks=[early_callback, wandb_callback],
                        callbacks=[reduce_lr, wandb_callback],  # other callback?
                        )

    print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))
    print('now is:  ' + now)
    # model.save(now + '.h5')

    model_path = '/root/autodl-tmp/vision_transformer/saved_model_' + now
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save(model_path + '.h5')


if __name__ == '__main__':
    main()
    # model = vit_large_patch16_224_in21k(num_classes=29, has_logits=False)
    # model.build((1, 224, 224, 3))
    # # model.summary()
    # for layer in model.layers:
    #     print(layer.name)
    #
    # print(111)
