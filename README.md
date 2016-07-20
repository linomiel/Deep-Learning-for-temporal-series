# Deep-Learning-for-temporal-series
This repository contain the main programs from my internship at IRISA in May to July 2016 under the supervision of Romain Tavenard, assistant professor at university of Rennes 2. 

The files operate in a given structure. They are supposed to be placed at the root of a repository, in wich each dataset has a subdirectory. Each dataset subdirectory is then split in experience subdirectory and each experience directory is finaly split in 2nd layer tuning directory.
Indeed more than one architecture can be trained given a common first layer.
-----------------------------------------------------------------------------------
The files are as follows:

brazilian_data.lua is converting the .txt files created with R into a format lua can use. It creates the 

classes<k>.data and dataset.data

model.lua is creating and training the neural network, and then saving it in model1.data. It calls 
autoencoder.lua and autoencoder2.lua .

autoencoder.lua trains the first convolutional layer. It produces error.txt and saves the last AE under 
ae_<k>.data where k is the number of the last iteration.

autoencoder2.lua is the same as autoencoder.lua but for the second convolution. It creates error2.txt and ae2_<k>.data .

model2.lua calls autoencoder2.lua and trains the second convolutional layer, using the results of the first one from model1.data . It is designed to finish model.lua's job if this one has been interrupted before its end. It creates model2.data

model_recovery.lua creates model2.lua if it does not already exists, using the best ae2 file it has.

printer.lua creates treated_patches.txt and target_patches.txt using the final data, wich can be used by python to train SVMs. treated_patches is a list of patches, and treated patches the corresponding class labels.

pre_printer.lua does the same as printer_lua, but on the raw data.