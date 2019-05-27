import cv2
import os


def augment_func(image_array, filename, output_path):

    global_counter = 0
    label = ['noise', 'gray', 'dim', 'mirror','color']    
    noise_kernel = [3,5]

    # Writing base image to augmentation directory

    cv2.imwrite(os.path.join(output_path, filename+'.jpg'), image_array)

    # Performing augmentation on mislabelled images            
    for value in label: # 1 augmentation
        if value == 'gray':            
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            file = os.path.join(output_path, str(filename) + '#' + str(global_counter) +'.jpg')
            cv2.imwrite(file, image)
            global_counter +=  1

        if value == 'color': # 3 augmentation
            col = ['blue','green','red']
            lst = [10, 10,10]
            for index, item in enumerate(col):

                image_array[:,:,index] = image_array[:,:,index] - lst[index]
                file = os.path.join(output_path, str(filename) + '#' + str(global_counter) + '.jpg')
                cv2.imwrite(file, image_array)

                global_counter += 1

        if value == 'noise': # 2 augmentation
            for k in noise_kernel:
                image = cv2.GaussianBlur(image_array, (k,k), 5)
                file = os.path.join(output_path, str(filename) + '#' + str(global_counter) + '.jpg')
                cv2.imwrite(file, image)
                global_counter += 1

        if value == 'dim': # 3 augmentation
            dim = [0.3,0.4,0.5]
            for i in range(len(dim)):
                img_dim = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCR_CB)
                img_dim[:,:,0] = img_dim[:,:,0] * dim[i]
                img_dim = cv2.cvtColor(img_dim, cv2.COLOR_YCR_CB2BGR)
                file = os.path.join(output_path, str(filename) + '#' + str(global_counter) +'.jpg')
                cv2.imwrite(file, img_dim)
                global_counter += 1

        # if value == 'mirror':
        #     flip = [0]
        #     for i in range(len(flip)):
        #         flip = cv2.flip(image_array, i)
        #         file = os.path.join(output_path, str(filename) + '#' + str(global_counter) + '.jpg')
        #         cv2.imwrite(file, flip)

        #         global_counter += 1
        

master_path = ['dataset/original/training/chairs', 'dataset/original/training/curtains', 'dataset/original/training/sofas', 'dataset/original/training/wardrobes']
output_path = ['dataset/augmented/chairs', 'dataset/augmented/curtains', 'dataset/augmented/sofas', 'dataset/augmented/wardrobes']

for index in range(1,4):
    for img in os.listdir(master_path[index]):
        if img.lower().endswith('.jpg'):
            image_array = cv2.imread(os.path.join(os.getcwd(),master_path[index],img))
            augment_func(image_array, img.split('.')[0], output_path[index])
            print('Processed {} Image'.format(img))
    print('************************************Processed {} dataset*********************************************'.format(master_path[index]))

