import matplotlib.pyplot as plt  
from PIL import Image
from tqdm import tqdm

import tensorflow as tf  
import numpy as np  
import os, sys, shutil
import math

def get_train_images(path):
    '''
    Get train image list from directory "path/1", "path/2", ...
    Here 1, 2, ... are different image classes.
    '''

    NUM_CLASSES = 0
    NUM_IMAGES = 0
    image_train, label_train = [], []
    for label in os.listdir(path):
        if label == '.DS_Store':
            continue
        for image in os.listdir(path+"/"+label):
            if image == '.DS_Store':
                continue
            image_train.append(path+"/"+label+'/'+image)
            label_train.append(label)
            NUM_IMAGES += 1
        NUM_CLASSES += 1
    train_set = np.array([image_train, label_train])
    train_set = train_set.transpose()
    
    np.random.shuffle(train_set) # shuffle
    image_list, label_list = list(train_set[:,0]), list(train_set[:,1])
    label_list = [int(i) for i in label_list]
    
    return image_list, label_list, NUM_IMAGES, NUM_CLASSES

def get_test_images(path):
    '''
    get test image list from directory "path"
    '''

    NUM_IMAGES = 0
    image_list = []
    for image in os.listdir(path):
        if image == '.DS_Store':
            continue
        image_list.append(path+"/"+image)
        NUM_IMAGES += 1

    return image_list, NUM_IMAGES

def load_image(path, train_or_test, train_scale=0.7):
    '''
    load images and preprocess them, like: resizing, normalization
    e.g. train_scale: 0.7, so train_set:test_set = 0.7:0.3
    '''
    if train_or_test == 'train':
        image_list, label_list, NUM_IMAGES, NUM_CLASSES = get_train_images(path)
    else:
        image_list, NUM_IMAGES = get_test_images(path)

    H, W, C = 64, 64, 3 # TODO: bigger or smaller
    X = np.random.randint(0, 256, (NUM_IMAGES,H,W,C), dtype=int) # random image
    with tf.Session() as sess: 
        for i in tqdm(range(NUM_IMAGES)): # or: trange(NUM_IMAGES)     
            try:
                image = Image.open(image_list[i]) # Exception: bad image or a file of some other kind
                if image.format == 'GIF':
                    image.seek(0) # just use the first frame
                if image.mode != 'RGB':
                    image = image.convert('RGB') # 3 channels
                resized = image.resize((H, W)) # TODO: or clip it to the proper size
                resized = np.array(resized)
                X[i,:,:,:] = resized[:,:,:] / 255. # normalization
            except Exception as e:
                print(image_list[i] + ": load error")
                # still use the random image: let it be
            else:
                pass
            finally:
                pass

    TRAIN_SIZE = int(train_scale * NUM_IMAGES)
    X_train, X_test = X[:TRAIN_SIZE,:,:,:], X[TRAIN_SIZE:,:,:,:]
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))

    if train_or_test == 'train':
        # generate labels Y
        Y = np.array(label_list)-1 # 1->0, 2->1, ...
        Y = Y.reshape([NUM_IMAGES,1])
        # use one-hot coding
        y_one_hot = tf.one_hot(Y, depth=NUM_CLASSES, axis=-1)
        with tf.Session() as sess:
            Y = sess.run(y_one_hot)
        Y = Y.reshape([NUM_IMAGES, NUM_CLASSES])

        Y_train, Y_test = Y[:TRAIN_SIZE,:], Y[TRAIN_SIZE:,:]
    else:
        Y_train, Y_test = None, image_list
    
    return X_train, Y_train, X_test, Y_test

def create_placeholders(H, W, C, NUM_CLASSES):
    '''
    Create placeholders for X, Y
    '''
    X = tf.placeholder(tf.float32, shape=(None, H, W, C))
    Y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
    
    return X, Y

def create_parameters():   
    '''
    Initialize parameters W1, W2
    '''                       
    W1 = tf.get_variable('W1', shape=(4,4,3,8), initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable('W2', shape=(2,2,8,16), initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1, "W2": W2}
    return parameters

def build_network(X, parameters, NUM_CLASSES):
    '''
    conv -> relu -> max_pool -> conv -> relu -> max_pool -> flatten -> fc
    '''

    W1, W2 = parameters['W1'], parameters['W2']
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') # CONV
    A1 = tf.nn.relu(Z1) # RELU
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME') # MAXPOOL
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME') # CONV
    A2 = tf.nn.relu(Z2) # RELU
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME') # MAXPOOL
    P2 = tf.contrib.layers.flatten(P2) # FLATTEN
    Y_PRED = tf.contrib.layers.fully_connected(P2, num_outputs=NUM_CLASSES, activation_fn=None) # FC
    
    return Y_PRED

def generate_random_batches(X, Y, batch_size=16): 
    NUM_IMAGES = X.shape[0] # number of training examples 
    batches = [] 
    # Step 1: Shuffle (X, Y) 
    permutation = list(np.random.permutation(NUM_IMAGES))
    shuffled_X = X[permutation,:,:,:] 
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case. 
    num_complete_batches = math.floor(NUM_IMAGES/batch_size) # number of batches of size batch_size in your partitionning 
    for k in range(0, num_complete_batches): 
        batch_X = shuffled_X[k*batch_size:k*batch_size+batch_size, :,:,:] 
        batch_Y = shuffled_Y[k*batch_size:k*batch_size+batch_size, :] 
        batch = (batch_X, batch_Y)
        batches.append(batch) 
     
    # Handling the end case (last batch < batch_size) 
    if NUM_IMAGES % batch_size != 0: 
        batch_X = shuffled_X[num_complete_batches*batch_size:NUM_IMAGES, :,:,:] 
        batch_Y = shuffled_Y[num_complete_batches*batch_size:NUM_IMAGES, :] 
        batch = (batch_X, batch_Y) 
        batches.append(batch) 
     
    return batches

def loss_f(Y_PRED, Y):
    '''
    loss function
    '''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_PRED, labels=Y))
    
    return loss

def model(X_train, Y_train, X_test, Y_test, learning_rate=1e-3, num_epochs=300, batch_size=16,
          load_model=False, model_path=None, model_name=None):
    """    
    Arguments:
    X_train -- training set, of shape (None, H, W, C)
    Y_train -- test set, of shape (None, NUM_CLASSES)
    X_test -- training set, of shape (None, H, W, C)
    Y_test -- test set, of shape (None, NUM_CLASSES)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    batch_size -- size of a batch
    load_model -- load model that already exists or create a new one
    model_path -- model_path to save model if model_path is not None else not save
    model_name -- model_name to save model if model_name is not None else not save
    
    Returns:
    parameters -- parameters learnt by the model.
    """
    
    tf.reset_default_graph()
    NUM_TRAIN_IMAGES, H, W, C = X_train.shape
    losses = []                                        # To keep track of the loss
    
    NUM_CLASSES = Y_train.shape[-1]
    X, Y = create_placeholders(H, W, C, NUM_CLASSES)
    parameters = create_parameters()
    Y_PRED = build_network(X, parameters, NUM_CLASSES)
    loss = loss_f(Y_PRED, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        if load_model:
            # saver = tf.train.import_meta_graph('%s-%d.meta' % (model_path+"/"+model_name, load_epoch=xxx))
            # but we have built the model already
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            # Run the initialization
            sess.run(init)
        
        for epoch in range(num_epochs):

            sum_batch_loss = 0. # train in batches
            num_batches = int(NUM_TRAIN_IMAGES / batch_size) # number of batches of size batch_size in the train set
            batches = generate_random_batches(X_train, Y_train, batch_size)
            for (batch_X, batch_Y) in batches:
                _, batch_loss = sess.run([optimizer, loss], feed_dict={X: batch_X, Y: batch_Y})
                sum_batch_loss += batch_loss
            
            average_batch_loss = sum_batch_loss / num_batches
            losses.append(average_batch_loss)
            if epoch % 10 == 0: # TODO: more or less rounds
                print ("Epoch %i: loss = %f" % (epoch, average_batch_loss))
                if model_path is not None:
                    saver.save(sess, model_path+"/"+model_name, global_step=epoch)
        
        # plot the loss
        plt.plot(np.squeeze(losses))
        plt.xlabel('iterations (per 10 epochs)')
        plt.ylabel('loss')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        label_pred_op = tf.argmax(Y_PRED, 1)
        correct_prediction = tf.equal(label_pred_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return parameters

def test(X_test, NUM_CLASSES, model_path, image_list, dstpath):
    '''
    predict images in [X_test, image_list]
    And move them to directory "dstpath/1", "dstpath/2", ... in turn. Here 1, 2, ... are different image classes.
    Also, we load model from model path.
    '''
    tf.reset_default_graph()
    
    _, H, W, C = X_test.shape
    X, Y = create_placeholders(H, W, C, NUM_CLASSES)
    parameters = create_parameters()
    Y_PRED = build_network(X, parameters, NUM_CLASSES)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('%s-%d.meta' % (model_path+"/"+model_name, load_epoch=xxx))
        # but we have built the model already
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # Calculate the correct predictions
        label_pred_op = tf.argmax(Y_PRED, 1)

        label_pred = label_pred_op.eval({X: X_test})
        for i in range(len(label_pred)):
            label, srcfile = label_pred[i], image_list[i]
            _, filename = os.path.split(srcfile) # get rid of directory before
            dstpath_class = dstpath + "/" + str(label+1) # 0->"1", 1->"2", ...
            if not os.path.exists(dstpath_class):
                os.makedirs(dstpath_class) # create directory
            dstfile = dstpath_class + "/" + filename
            shutil.move(srcfile, dstfile)
            # shutil.copyfile(srcfile, dstfile)

# main module:
# X_train, Y_train, X_test, Y_test = load_image(path="train", train_or_test='train', train_scale=0.7)
# if first train:
# parameters = model(X_train, Y_train, X_test, Y_test, learning_rate=1e-3, num_epochs=101, 
#     load_model=False, model_path="model", model_name="classifier")
# if continue train:
# parameters = model(X_train, Y_train, X_test, Y_test, learning_rate=2e-4, num_epochs=301, 
    # load_model=True, model_path="model", model_name="classifier")
    
# if test:
NUM_CLASSES = 2
_, _, X_test, image_list = load_image(path="test", train_or_test='test', train_scale=0)
test(X_test, NUM_CLASSES, model_path="model", image_list=image_list, dstpath="test_out")
