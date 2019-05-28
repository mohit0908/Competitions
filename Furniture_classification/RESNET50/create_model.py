import tensorflow as tf
import os


classes = ['chairs', 'curtains','sofas','wardrobes']



def create_model(checkpoint_file):

        model = tf.keras.applications.ResNet50(include_top = True, weights ='imagenet')

        num_classes = len(classes)

        last_layer = model.layers[-2].output # grabing 2nd last layer output
        out = tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
        custom_resnet_model = tf.keras.models.Model(inputs = model.input, outputs = out)
        for layer in custom_resnet_model.layers[:-2]:
                layer.trainable  = False

        # Load old weights if present
        print('Exists:', os.path.isfile(checkpoint_file))
        try:
            if os.path.isfile(checkpoint_file):
                custom_resnet_model.load_weights(checkpoint_file)
                print('Pretrained weights loaded from ckpt')
        except:
            pass

        custom_resnet_model.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
        print('Custom resnet model created')

        return custom_resnet_model

