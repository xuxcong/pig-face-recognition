from resvgg_model import *;
from read_data import *;


if __name__ == '__main__':
    dataset_path = '/home/smie/zhengjx/Res_Bilinear_cnns/data_model/';
    model_path = '/home/smie/zhengjx/Res_Bilinear_cnns/data_model/';
    save_model_best = model_path + 'res_fine_last_layers_epoch_best.npz';
    save_model_last = model_path + 'res_fine_last_layers_epoch_last.npz';
    train_data_file = [];
    val_data_file = [];
    #define the data to use
    for i in range(0,50):
        train_data_file.append(dataset_path + 'train_data' + str(i) + '.h5');
        val_data_file.append(dataset_path + 'validation_data' + str(i) + '.h5');
    num_class = 30;
    train_batch_size = 8;
    val_batch_size = 8;
    train_reader = data_reader(train_data_file, num_class, train_batch_size);
    val_reader = data_reader(val_data_file, num_class, val_batch_size)

    model_path =  model_path + 'res_fine_last_layers_epoch_best.npz';
    
    sess = tf.Session()     ## Start session to create training graph
    keep_prob = tf.placeholder(tf.float32);
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 30])

    # resvgg = resvgg(imgs, 'vgg16_weights.npz', sess, finetune = False)   # fine tuning
    resvgg = resvgg(imgs, keep_prob, 'vgg16_weights.npz', sess)
    
    print('Res Bilinear cnn network created')
    
    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=resvgg.fc3l, labels=target))

    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.2, momentum=0.4).minimize(loss)  # for fine tuning
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)        # for normal training
    check_op = tf.add_check_numerics_ops()


    correct_prediction = tf.equal(tf.argmax(resvgg.fc3l,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    #resvgg.load_initial_weights(sess)
    resvgg.load_own_weight(sess , model_path);   # load the model trained before 

    
        


    correct_val_count = 0
    val_loss = 0.0
    while(val_reader.have_next()):
        batch_val_x, batch_val_y = val_reader.next_batch();
        val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y, keep_prob:1.0})
        pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y, keep_prob:1.0})
        correct_val_count+=pred
    val_loss = val_loss/(1.0*val_reader.total_datanum);
    print("##############################")
    print("Validation Loss -->", val_loss)
    print("correct_val_count, total_val_count", correct_val_count, val_reader.total_datanum)
    print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*val_reader.total_datanum))
    print("##############################")



    print('Starting training')
    best_validation_lost = val_loss;
    for epoch in range(50):
        train_reader.new_iterator();
        ave_cost = 0;
        num = 100;
        i = 0;
        while(train_reader.have_next()):
            i = i + 1;
            batch_xs, batch_ys = train_reader.next_batch(process = True);
            start = time.time()
            sess.run([optimizer,check_op], feed_dict={imgs: batch_xs, target: batch_ys, keep_prob:0.5})
            cost = sess.run(loss, feed_dict={imgs: batch_xs, target: batch_ys, keep_prob:1.0})
            ave_cost = ave_cost + cost;
            if i % num == 0:
                ave_cost = 1.0 * ave_cost / num;
                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,"Loss:", str(ave_cost))
                ave_cost = 0;


        correct_val_count = 0
        val_loss = 0.0;
        val_reader.new_iterator();
        while(val_reader.have_next()):
            batch_val_x, batch_val_y = val_reader.next_batch();
            val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y, keep_prob:1.0})
            pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y, keep_prob:1.0})
            correct_val_count+=pred
        val_loss = val_loss/(1.0*val_reader.total_datanum);
        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, val_reader.total_datanum)
        print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*val_reader.total_datanum))
        print("##############################")
        #save the best model
        if(val_loss < best_validation_lost):
            best_validation_lost = val_loss;
            last_layer_weights = []
            for v in resvgg.parameters:
                last_layer_weights.append(sess.run(v))
            np.savez(save_model_best,last_layer_weights)
            print('save the model!')


    last_layer_weights = []
    for v in resvgg.parameters:
        print(v)
        last_layer_weights.append(sess.run(v))
    np.savez(save_model_last,last_layer_weights)
    print('save the model!')
