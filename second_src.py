import keras
from keras import layers,models,utils,optimizers
import matplotlib.pyplot as plt
import numpy as np

# Processing Data

print("train data:")
train_data=utils.image_dataset_from_directory(
    'C:/Users/User/Desktop/machinlearning/vscode scripts/mnist-mlp practice/human fealings/new_dataset2/train',
    image_size=(48,48),
    batch_size=128,
    label_mode="categorical"
)

print("val data:")
val_data=utils.image_dataset_from_directory(
    'C:/Users/User/Desktop/machinlearning/vscode scripts/mnist-mlp practice/human fealings/new_dataset2/validation',
    image_size=(48,48),
    batch_size=128,
    label_mode="categorical"
)

print("test data:")
test_data=utils.image_dataset_from_directory(
    'C:/Users/User/Desktop/machinlearning/vscode scripts/mnist-mlp practice/human fealings/new_dataset2/test',
    image_size=(48,48),
    batch_size=128,
    label_mode="categorical"
)

# Visualize Sample Training Data

a,f,h,r,s=0,1,2,3,4
for data_batch,label_batch in train_data:
    print("data_batch shape:",data_batch.shape)
    fig,ax=plt.subplots(1,5,figsize=(15,15))
    for i,axe in enumerate(ax):
        title=''
        label=np.argmax(label_batch[i])
        if label==a:
            title='Angry'
        elif label==f:
            title='Fear'
        elif label==h:
            title='Happy'
        elif label==r:
            title='Sad'
        else:
            title='Suprise'
        axe.imshow(data_batch[i].numpy().astype('uint8'))
        axe.set_axis_off()
        axe.set_title(title)
    break
plt.show()

# Model Creation

data_augmentation=keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

input=keras.Input(shape=(48,48,3))
x=data_augmentation(input)
x=layers.Conv2D(32,(3,3),activation='relu')(input)
x=layers.BatchNormalization()(x)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Dropout(0.25)(x)
x=layers.Conv2D(64,(3,3),activation='relu')(x)
x=layers.BatchNormalization()(x)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Dropout(0.25)(x)
x=layers.Flatten()(x)
x=layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
x=layers.Dropout(0.5)(x)
output=layers.Dense(5,activation='softmax')(x)

model=keras.Model(inputs=input,outputs=output)
model.summary()

# Train and Evaluation

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_data,epochs=30,validation_data=val_data)

# Plot Losses And Accuracy

accuracy=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(accuracy)+1)
plt.plot(epochs,accuracy,'bo',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Val Accuracy")
plt.title("val and train accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label="loss")
plt.plot(epochs,val_loss,'b',label="val loss")
plt.title("train and val loss")
plt.legend()
plt.show()