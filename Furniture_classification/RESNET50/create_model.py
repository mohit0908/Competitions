import tensorflow as tf
import os


classes = ['chairs', 'curtains','sofas','wardrobes']
checkpoint_path = 'ckpt/*.hdf5'



def create_model():

        model = tf.keras.applications.ResNet50(include_top = True, weights ='imagenet')

        num_classes = len(classes)

        last_layer = model.layers[-2].output # grabing 2nd last layer output
        out = tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
        custom_resnet_model = tf.keras.models.Model(inputs = model.input, outputs = out)
        for layer in custom_resnet_model.layers[:-2]:
                layer.trainable  = False

        # Load old weights if present
        if os.path.isfile(checkpoint_path):
            custom_resnet_model.load_weights(checkpoint_path)
            print('Pretrained weights loaded from ckpt')

        custom_resnet_model.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
        print('Custom resnet model created')

        return custom_resnet_model