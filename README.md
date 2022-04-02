The purpose of the project are as follows,

1. Training CNN from Scratch(trainingFromScratch).
2. Fine tune a pretrained model on the iNaturalist dataset.
3. Use a pretrained Object Detection model for a social application

# Part A - Training CNN from Scratch

**train_data, val_data, test_data =datagen(batch_size, augment_data)**

Data preperation, Data splits and Data augmentation are being done inside this function.

**model = CNN_model(activation_func_convolution,activation_func_dense , no_of_filters, filter_convol_size, filter_pool_size, batch_normalisation, dense_layer, dropout ,image_size)**

This function builds the CNN model structure with the following parameters,
1. activation_func_convolution : Activation function to be used in convolution layers.         
2. activation_func_dense :  Activation function to be used at the dense layers.          
3. no_of_filters : Number of filters in first layer, number of filters for other layers are halved/squared/hybrid inside the CNN_model function.         
4. filter_convol_size : List of size of kernel in every layer.       
5. filter_pool_size : Pooling size. pool size is (2,2).        
6. batch_normalisation: Whether batch normalization has to be used or not. Boolean value. If true, batch normalization is done at every layer.       
7. dense_layer : Size of dense layers.       
8. dropout : The value of dropout which has to be used the dense layers.     
9. image_size : Dimension of the input.

**model_det = model.fit( train_data, steps_per_epoch = train_step_size, validation_data = val_data, validation_steps = val_step_size, epochs=20, 
          callbacks=[WandbCallback(data_type="image", generator=val_data), earlyStopping, mc], verbose=2 )**
      
Trains the model for a fixed number of epochs. Trains with the input data (train_data) and target data (val_data). It also has a callback parameter

**sweep**   
The sweep configuration allows us to run several number of experiments with the hyperparameters.    
The sweep calls the **model_train** function which comprises the Data preperation, Model creation and model fits. 
The best model and its hyperparameters could be obtained from the sweep.

**Best model performance:**

Test accuracy: 41.10 %

**Visualise filters and plotting 10 x 3 grid containing sample images with predictions**    
plotGrid(img, row, col) function is used to visualize the filters in the first layer.
The best model contains 64 filters in the first layer.
Visualized all the filters for a random image from the test dataset.
plotclass(samples, rows, cols) function is used to plot a 10 x 3 grid sample images from the test dataset with their predictions made.

**Guided backpropagation**    
A random image is taken and the plots are generated.
The generated plot shows which part of the image excites various neurons in the fifth convolution layer.

**Passing hyperparameters as command line arguments**     
You can pass the following command with the 'inaturalist_12K' data folder in the present working directory.

**Usage**
* ```no_of_filters```, this argument requires the number of filters to be used by the model.
* ```augment_data```, this argument requires True or False and decides whether to augment data or not.

python Part_A_cmdline.py 64 True 64 True 0.15 128 3   
where Part_A_cmdline.py is the filename and the parameters of order,    
python <filename> <no_of_filters> <augment_data> <batch_size> <batch_normalisation> <dropout> <dense_layer> <filter_convol_size> 



