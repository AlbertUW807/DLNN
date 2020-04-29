# DLNN
Deep Learning & Neural Networks Projects

# Projects
[Logistic Regression](https://github.com/AlbertUW807/DLNN/tree/master/Logistic%20Regression)
  - Implemented an Image Recognition Algorithm that recognizes cats with 67% accuracy!
  - Used a logistic regression model.

[Deep Learning Model](https://github.com/AlbertUW807/DLNN/tree/master/Deep%20Learning%20Model)
  - Implemented an Image Recognition Algorithm that recognizes cats with 80% accuracy!
  - Used a 2-layer neural network (LINEAR->RELU->LINEAR->SIGMOID) 
            and an L-layer deep neural network ([LINEAR->RELU]*(L-1)->LINEAR->SIGMOID).
  - Trained the model as a 4-layer neural network.

[Model Initialization](https://github.com/AlbertUW807/DLNN/tree/master/Model%20Initialization)
  - Implemented different initialization methods to see their impact on model performance (3-Layer).
  - Zero Initialization -> Fails to break symmetry (all parameters to 0).
  - Random Initialization -> Breaks symmetry, more efficient models.
  - He Initialization -> Xavier Initialization without scaling factor, recommended for layers with ReLU activation.
