from model import model
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers
from dataloader import train_iterator
import math



def train_step(model, images, labels,labellen,seqlen, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss = tf.nn.ctc_loss(labels=labels,logits=prediction,label_length=labellen,logit_length=seqlen,logits_time_major=False,blank_index=0)
        loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

def train(model, data_iterator, optimizer):

    for i in tqdm(range(int(10000/50))):
        images, labels,labellen,seqlen = data_iterator.next()
        loss, prediction = train_step(model, images, labels,labellen,seqlen, optimizer)

        print('loss: {:.6f}'.format(loss))

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)
    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

if __name__ == '__main__':
    train_data_iterator = train_iterator()

    model = model(input_shape=[1400, 80, 1])
    model.build(input_shape=(None,) + (1400,80,1))

    # model = tf.keras.models.load_model("./h5/pose-155.h5")
    model.load_weights("./h5/zw-5.h5")

    model.summary()


    optimizer = optimizers.Adam()

    # learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=0.005,
    #                                                 decay_steps=1000 * int(12000/50) - 20,
    #                                                 alpha=0.000001,
    #                                                 warm_up_step=20)
    # optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)


    for epoch_num in range(200):
        train(model, train_data_iterator, optimizer)
        if epoch_num%2==0:
            model.save('./h5/zw-%s.h5'%epoch_num, save_format='h5')

