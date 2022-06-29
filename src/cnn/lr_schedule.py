import tensorflow as tf

__reduce_on_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    mode="max",
    min_delta=0.0001
)

def __multi_step_lr_scheduler(epoch, lr):
    if epoch == 60 or epoch == 120 or epoch == 160:
        return lr/5
    else:
        return lr

def __multi_step_lr_scheduler_v2(epoch, lr):
    if epoch == 150 or epoch == 255:
        return lr/5
    else:
        return lr


__multi_step_lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(__multi_step_lr_scheduler)

__cb_map = {
    'reduce_on_plateau': __reduce_on_plateau_cb,
    'multi_step': __multi_step_lr_scheduler_cb,
    'multi_step_v2': __multi_step_lr_scheduler_v2
}

def resolve_schedular_callback(scheduler:str):
    if scheduler not in __cb_map:
        raise ValueError('invalid scheduler: "{}"'.format(scheduler)) 
    
    return __cb_map[scheduler]