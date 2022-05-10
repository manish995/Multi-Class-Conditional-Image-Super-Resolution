## Library versions 
torch=1.11.0   //GPU version , on cpu I don't tested probably not run on cpu.
torchvision=0.12.0
PIL=9.0.1
matplotlib=1.3.4
pandas=1.3.4



We used nvidia rtx3060ti 6GB graphic Card, Cse GPU servers , on 4 GB its will not run
\\As mookit donot support more than 250 mb size so  I upload all the models and data to drive and here are the link
\\Augmented data link: https://drive.google.com/file/d/1TcwYjj_JYQ68k7LBAV3oDMUSyg85kvYH/view?usp=sharing
\\MA link: https://drive.google.com/file/d/1NE9Hn-Z6X59PuYlBWAXq54jCigzpsDsq/view?usp=sharing
\\All model link : https://drive.google.com/drive/folders/1v1r9g5dPCcLUaQWjXDTo4f8dD2FjtuI6?usp=sharing

\\put Augmented_data folder under data folder discussed below.
\\put MA folder under data folder discussed below.
\\put all 4 generator under Demo/SRCGAN/saved_models
\\put gender classifier into Gender_results/SRCGAN
\\put eyeglasses classifier under Eyeglasses_result/SRCGAN and eye_hat_result/SRCGAN
\\put hat classifier under eye_hat_result/SRCGAN


## Folder Description
There are total 6 folders
1. data folder
It contain the augmented data folder and MA folder which have necessary data to train all the models.

2. ESRGAN-Results/SRCGAN
It contains all the files regarding the trainning of ESRGAN
model.py contain the model acrhitecture
dataset.py load the data
esrgan.py contain the training details
esrgan_result.csv contain the true label and predicted label on the generated image
Accuracy_calculator.ipynb contain code for calculating the confusion matrix for esrgan
saved_model folder will contain model saved during training 
images/training will contain images saved during training

To train the esrgan model : run python3 esrgan.py
//please note it will take some during start of training (2-3 min), some line may print in loop during that time.

3. Eyeglasses_result/SRCGAN
It contains all the files regarding the trainning of ESRGAN + Eyeglasses
model.py contain the model acrhitecture
classifier.py contain the classifier acrhitecture 
dataset.py load the data
esrgan.py contain the training details
esrgan_result_eye.csv contain the true label and predicted label on the generated image
Accuracy_calculator.ipynb contain code for calculating the confusion matrix for esrgan +Eyeglasses
saved_model folder will contain model saved during training 
images/training will contain images saved during training
real_eyeglasses_classifier.pt is saved eyeglasses classifier model

To train the esrgan+ Eyeglasses model : run python3 esrgan.py
//please note it will take some during start of training (2-3 min), some line may print in loop during that time.

4. Gender_results/SRCGAN
It contains all the files regarding the trainning of ESRGAN + Gender
model.py contain the model acrhitecture
classifier.py contain the classifier acrhitecture 
dataset.py load the data
esrgan.py contain the training details
esrgan_result_gender.csv contain the true label and predicted label on the generated image
Accuracy_calculator.ipynb contain code for calculating the confusion matrix for esrgan + Gender
saved_model folder will contain model saved during training 
images/training will contain images saved during training
our_gender_classifier.pt is saved gender classifier model

To train the esrgan+ Gender model : run python3 esrgan.py
//please note it will take some during start of training (2-3 min), some line may print in loop during that time.


5. eye_hat_result/SRCGAN
It contains all the files regarding the trainning of ESRGAN + Eyeglasses+ Hat
model.py contain the model acrhitecture
classifier.py contain the classifier acrhitecture 
dataset.py load the data
mccisr.py contain the training details
esrgan_result_eye_hat.csv contain the true label and predicted label on the generated image
Accuracy_calculator.ipynb contain code for calculating the confusion matrix for esrgan + Eyeglasses+ Hat
saved_model folder will contain model saved during training 
images/training will contain images saved during training
real_eyeglasses_classifier.pt is saved Eyeglasses classifier model
real_hat_classifier.pt is saved hat classifier model

To train the esrgan+ Gender model : run python3 mccisr.py
//please note it will take some during start of training (2-3 min), some line may print in loop during that time.


6. Demo/SRCGAN
It contain all the files for running the all 4 saved model on the images where esrgan performed poor but our model performed good to clearly see the difference. 
model.py contain the model acrhitecture
classifier.py contain the classifier acrhitecture 
dataset.py load the data
real_eyeglasses_classifier.pt is saved eyeglasses classifier model
our_gender_classifier.pt is saved gender classifier model
real_hat_classifier.pt is saved hat classifier model
list_attr_celeba.csv is the original csv which has all images with 40 attributes
Final_demo.py contain the code to test the model on images
Demo_images contain 3 folder:
   1. Eye: Contain all images for which esrgan images labelled with no eyeglasses but our model generated images is classified correctly.
   2. Eye_Hat: contain 2 folder (Eye , Hat) all images for which esrgan images labelled with wrong positive class but our model generated images is classified correctly.
   3. Gender: It contain all the images for which esrgan images classifier incorrectly but our model generated images is classified correctly.
saved_model contain all 4 generator model weights which are used for generating images.
images/output will contain output after running the Final_demo.py

To run Final_demo.py : python3 Final_demo.py
p,q,r in Final_demo.py in line number77,78,79 (variable will help to control the output images you want to get for Gender Eyeglasses (Eyeglasses+Hat) respectively)







