import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


image_dir=r"C:\Users\Gayathri Anil\Downloads\cropped"
virat_images=os.listdir(image_dir+ '\\virat_kohli')
serena_images=os.listdir(image_dir+ '\\serena_williams')
roger_images=os.listdir(image_dir+ '\\roger_federer')
maria_images=os.listdir(image_dir+ '\\maria_sharapova')
lionel_images=os.listdir(image_dir+ '\\lionel_messi')


print("--------------------------------------\n")

print('The length of Virat Kohli images is',len(virat_images))
print('The length of Serena Williams images is',len(serena_images))
print('The length of Rodger Federer images is',len(roger_images))
print('The length of Maria Sharapova images is',len(maria_images))
print('The length of Lionel Messi images is',len(lionel_images))



print("--------------------------------------\n")
dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(virat_images),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(serena_images),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)
        
for i , image_name in tqdm(enumerate(roger_images),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)
        
        
for i ,image_name in tqdm(enumerate(maria_images),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)        
        
for i ,image_name in tqdm(enumerate(lionel_images),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)        
              
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")


# Normalizing the Dataset
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Correcting the labels if needed (Ensure they range from 0 to 4 for 5 classes)
# label = label - 1  # If labels don't start from 0

# Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Change to 5 neurons for 5 classes
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Change loss function
              metrics=['accuracy'])

# Training the Model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.3)

# Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report:\n', classification_report(y_test, y_pred))



# Load and preprocess a single image
def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

images = [r"C:\Users\Gayathri Anil\Downloads\cropped\maria_sharapova\maria_sharapova7.png",
                         r"C:\Users\Gayathri Anil\Downloads\cropped\virat_kohli\virat_kohli5.png",
                         r"C:\Users\Gayathri Anil\Downloads\cropped\serena_williams\serena_williams6.png",
                         r"C:\Users\Gayathri Anil\Downloads\cropped\roger_federer\roger_federer19.png",
                         r"C:\Users\Gayathri Anil\Downloads\cropped\lionel_messi\lionel_messi39.png"]

# Preprocess the single image
for i in images:
    single_image = preprocess_single_image(i)

    # Reshape the image to fit the model's input shape
    single_image = np.expand_dims(single_image, axis=0)

    # Make predictions using the model
    predictions = model.predict(single_image)
    predicted_class = np.argmax(predictions)

    class_names = ['Virat Kohli', 'Serena Williams', 'Roger Federer', 'Maria Sharapova', 'Lionel Messi']
    predicted_label = class_names[predicted_class]

    print(f"The predicted label for the image is: {predicted_label}")


#During the data processing phase, we used for loops to iterate through the each players and it checks if the image is png and then loads, coverts to RGB, resizes, and appends it to the dataset along with its respective label.
#During the dataset preparation process, we converts the dataset to numpy arrays which is compatible with Tensorflow.
#The chosen model is simple CNN model architecture. It is designed for image classification tasks which gives better performance among other models. 
#During the training process, we set the number of epoch as 50 and batch size as 32. Also, we have set the validation split as 0.3 so that it reserves a portion of the training data as validation set.
#During model evaluation, we get the classification report showing precision, recall, F1-score and accuracy of each class. We got an accuracy of 86.27%.
#we have also printed the preditions for each class.
