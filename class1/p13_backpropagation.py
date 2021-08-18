import tensorflow as tf
w = tf.Variable(tf.constant(5,dtype=tf.float32))
# tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
# tf.constant()  创建常量
lr = 0.2  #可以换为0.01等等
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w+1)
    grads = tape.gradient(loss,w)

    w.assign_sub(lr*grads)
    print("训练第%s次之后，w是%f，损失值是%f" % (epoch, w.numpy(), loss))

