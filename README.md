# signalproject
 Optimisation of self-labeling-refinement pipeline on EEG classification model for Harmful Brain Activity Classification competition

Starting off, the code provides a comprehensive setup for training a machine learning model, including defining constants, necessary libraries, 
configuring the environment for GPU usage, setting random seeds for reproducibility, and preparing the training data for the specific task at hand. 

Then, the code sets up a robust framework for training a neural network model for a classification task using TensorFlow/Keras. 

Next, the cross_validate_model function performs cross-validation for training and evaluating models, ensuring robustness by handling different data qualities 
and maintaining data integrity throughout the process. It trains two models sequentially within each fold, incorporates pseudo labeling for data augmentation, 
and computes evaluation metrics for performance assessment.

After that, these functions facilitate the creation and processing of spectrograms from EEG data, crucial for analyzing brain activity patterns. 
The generated spectrogram images are then utilized for further analysis and classification tasks, such as identifying harmful brain activity. 

Finally, the code prepares a data generator class for handling EEG data and corresponding labels. It then employs this generator to make predictions using a pre-trained 
neural network model for test data. Finally, it aggregates predictions across multiple folds and saves them into a submission file for further evaluation.
