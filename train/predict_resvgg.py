from resvgg_model import *;
from read_data import *;

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

if __name__ == '__main__':

    dataset_path = '/home/smie/zhengjx/Res_Bilinear_cnns/data_model/';
    model_path = '/home/smie/zhengjx/Res_Bilinear_cnns/data_model/';
    save_model_best = model_path + 'res_fine_last_layers_epoch_best.npz';
    save_model_last = model_path + 'res_fine_last_layers_epoch_last.npz';
    test_data_file = [dataset_path + 'testB.h5'];
    val_data_file = [dataset_path + 'new_val_448.h5'];
    num_class = 30;
    test_batch_size = 1;
    val_batch_size = 1;
    test_reader = data_reader(test_data_file, num_class, test_batch_size, shuffle = False);
    val_reader = data_reader(val_data_file, num_class, val_batch_size)

    model_path =  model_path + 'res_fine_last_layers_epoch_best.npz';
    
    sess = tf.Session()     ## Start session to create training graph
    keep_prob = tf.placeholder(tf.float32);
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 30])

    # resvgg = resvgg(imgs, 'vgg16_weights.npz', sess, finetune = False)
    resvgg = resvgg(imgs, keep_prob, 'vgg16_weights.npz', sess, res = False)
    
    print('VGG network created')
    
    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=resvgg.fc3l, labels=target))

    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.0005, momentum=0.4).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)
    check_op = tf.add_check_numerics_ops()


    correct_prediction = tf.equal(tf.argmax(resvgg.fc3l,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    #resvgg.load_initial_weights(sess)
    resvgg.load_own_weight(sess , model_path);

    
    # Use the validation loss to make sure that we have loaded the right model
    correct_val_count = 0
    val_loss = 0.0
    while(val_reader.have_next()):
        batch_val_x, batch_val_y = val_reader.next_batch();
        val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y, keep_prob: 1.0})
        pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y, keep_prob: 1.0})
        correct_val_count+=pred
    val_loss = val_loss/(1.0*val_reader.total_datanum);
    print("##############################")
    print("Validation Loss -->", val_loss)
    print("correct_val_count, total_val_count", correct_val_count, val_reader.total_datanum)
    print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*val_reader.total_datanum))
    print("##############################")

    #parameter for test
    target_path = '/home/smie/zhengjx/Res_Bilinear_cnns/train_test/testB.txt';
    images = []
    with open(target_path, 'r') as f:   
        for l in f.readlines():
            l = l.strip('\n').split()
            name = l[0].split('/')[-1].split('.')[0];
            images.append(name)
    csvfile = file('b_cnn_' + 'test' +'.csv', 'wb')
    writer = csv.writer(csvfile)
    i = 0;
    while(test_reader.have_next()):
        batch_test_x, batch_val_y = test_reader.next_batch();
        result = sess.run([resvgg.fc3l], feed_dict={imgs: batch_test_x, keep_prob: 1.0});      
        result = softmax(result);
        if(i % 100 == 0):
            print(i)
        for j in range(0,30):
            writer.writerow([images[i], j + 1, max(round(result[0][0][j],7) - 0.000001, 0.0) * 0.96 + 0.001333 ])
        i = i + 1;
    csvfile.close();