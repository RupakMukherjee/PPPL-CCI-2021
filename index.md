
| Equation| Equation Graph| Validation Loss Graph| Error Graph|
|:------:|:------:|:-------------------:| :---------:|
|Advection|<img width="250" alt="Screen Shot 2021-10-31 at 9 01 22 PM" src="https://user-images.githubusercontent.com/90737587/139620001-ab139012-a904-4bce-8c1a-660ef648a118.png">|<img width="250" alt="A5VL" src="https://user-images.githubusercontent.com/90737587/139617653-78702a5e-3fa7-4e3a-8ab2-681175bf3d65.png">|<img width="420" alt="A5EC" src="https://user-images.githubusercontent.com/90737587/139617664-9f2e7caa-e898-492e-b16c-45c970dc4df8.png">|
| Soliton|<img width="250" alt="Screen Shot 2021-10-31 at 8 30 29 PM" src="https://user-images.githubusercontent.com/90737587/139617944-881010bb-8643-42a4-947d-4a9a221482c7.png">|<img width="250" alt="Test4-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139618438-a89322c0-8d2b-48de-b0ec-d7141f93a2af.png">|<img width="425" alt="Test5-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139618449-d72bf16a-d886-4e0e-b21b-9b03fbd37fa9.png">|
|Burger||<img width="250" alt="TestD-ValLoss (5)" src="https://user-images.githubusercontent.com/90737587/139616870-89b34776-672d-48fc-aaa7-d3ea45dfdff6.png">|<img width="420" alt="TestD-Error (4)" src="https://user-images.githubusercontent.com/90737587/139617121-9dcdd70e-8846-4029-aa1a-be35c2ed7367.png">|








In the beginning of this internship, I learned to solve math computations in programming languages: Python, C++, and Fortran. First, I learned how to operate the Mac terminal and downloaded the compatible text editors to run the programs. My first assignment was to write a program to calculate the sum of two numbers. For this assignment, I learned the basic syntax rules, variables, inputs, outputs, and how to utilize the error codes to debug my programs. My second assignment was to create a program to calculate the sum of numbers from 0 to 100. For this assignment, I learned how to use loops. My third assignment was to create a program to solve the problem: a point particle moving in the x-direction with a constant velocity. For this assignment, I learned how to insert variables in loops. My fourth assignment was to develop a program to solve the equation u(i+1)=u(i)+1*dx utilizing arrays. For this assignment, I learned how to write an array with a loop inside and align the array by the decimal points. After I got familiar with creating programs to solve more complex math equations, I was introduced to the machine learning process and TensorFlow. 

TensorFlow is an open-source library used for machine learning and artificial intelligence. I studied a Python code that solved the Advection Diffusion equation from GitHub. The machine learning process begins by entering data that will affect the rest of the code. Then, we must train the code in a series of ways to get the correct output. This way contains a series of tests until the output is the smallest margin of error possible. The series of tests does not always depend on the tests before it. In order to run it on my computer, I needed to install correct import such as TensorFlow and Keras.  After I was able to run the Advection Diffusion equation, I changed the equation to the Burger’s equation but the same machine learning format. 

I changed the parameters of the sequential model of the Burger’s equation to try to get closer to the correct output. I learned that by changing the parameters multiple times on the first test of Burger’s equation code, I was able to reduce the validation loss but only slightly.  I still needed to run a second test to get the best results.  After changing parameters a few times with the second test of the Burger’s equation code, it did not make a big difference on the output and graph.  Each time the output was very accurate, the dotted line and solid line were aligned, which meant the second test was better than the first despite multiple attempts to reduce validation loss.  I created a table of various inputs and outputs of the Burger’s equation. I created create tables to display the various inputs of parameters and output graphs for the Burger’s Equation on LaTex. I learned how to format tables on LaTeX and inserts images. I studied how the inputs affected the graphs by researching about Keras dense layers and sequential models. For Test 1, I noticed that the errors decrease when I increased  the  parameters and epoch numbers.  However, Test 1 results is not as accurate as Test 2.  For Test 2, despite various changes to inputs, Test 2 has very low errors. 

## Burger's Equation

### Test 1: Input code 
```
# Build model
deep_approx = keras.models.Sequential()
deep_approx.add(layers.Dense(10, input_dim=2, activation='elu'))
deep_approx.add(layers.Dense(10, activation='elu'))
deep_approx.add(layers.Dense(1, activation='linear'))

# Compile model
deep_approx.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_approx.fit(X_train, y_train,
            epochs=10, batch_size=32,
            validation_data=(X_dev, y_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
            
deep_approx.summary()
```

### Test 1: Input Table 
| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Epochs|
|:---:|:------------: | :-----------: |:------------:|:---:|
| A|10|10|1|10|
| B|2|10|1|30|
| C|4|4|5|30|
| D|15|15|2|30|
| E|50|50|1|30|

### Test 1: Output Graphs 
| Test| Validaiton Loss Chart| Error Chart|
|:---:|:------------: |:-----------: |
|A|<img width="350" alt="TestA-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619306-480ea9e0-f654-4864-bda3-ae5a0f53c418.png">|<img width="550" alt="TestA-Error (1)" src="https://user-images.githubusercontent.com/90737587/139619329-5b7955f1-579b-4ee3-a147-bd3197126564.png">|
|B|<img width="350" alt="TestB-ValLoss (4)" src="https://user-images.githubusercontent.com/90737587/139619412-95f597ab-cbfc-4063-8eaa-30f4491a43e8.png">|<img width="550" alt="TestB-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139619425-680dee6c-396c-4ae8-a8cd-bcb17c41b372.png">|
|C|<img width="350" alt="TestC-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619474-6810223d-fcfc-436e-a9f0-3b4386f0b234.png">|<img width="550" alt="TestC-Error (1)" src="https://user-images.githubusercontent.com/90737587/139619498-d43d8a6b-786e-4d54-bcdf-3486bd74d326.png">|
|D|<img width="350" alt="TestD-ValLoss (6)" src="https://user-images.githubusercontent.com/90737587/139619635-e958cf35-bacd-4a92-a7cf-759393356139.png">|<img width="550" alt="TestD-Error (5)" src="https://user-images.githubusercontent.com/90737587/139619653-7c44d6e6-4caa-433b-91ea-e701b2f2ccad.png">|
|E|<img width="350" alt="TestE-VaLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619684-dc647f49-c17e-4137-820a-4be6be4dfa66.png">|<img width="550" alt="TestE-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139619696-ffe55108-72fa-454e-ae4b-9ecc3a61a288.png">|











<img width="900" alt="Screen Shot 2021-10-25 at 11 42 01 PM" src="https://user-images.githubusercontent.com/90737587/138822525-54a3ab13-b541-42e8-b6f1-61debbb85c63.png">

### Test 2: Input Code
```
# Build model
deep_stepper2 = keras.models.Sequential()
deep_stepper2.add(layers.Dense(2, input_dim=6, activation='elu'))
deep_stepper2.add(layers.Dense(10, activation='elu'))
deep_stepper2.add(layers.Dense(1, activation='linear'))

# Compile model
deep_stepper2.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_stepper2.fit(Xs_train, ys_train, epochs=30, batch_size=32,
            validation_data=(Xs_dev, ys_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
            
```
### Test 2: Input Table
<img width="900" alt="Screen Shot 2021-10-25 at 11 44 18 PM" src="https://user-images.githubusercontent.com/90737587/138822811-f09bf4b3-33f1-4ba9-b3fa-1fe5e1f4768e.png">

### Test 2: Output Graphs
<img width="900" alt="Screen Shot 2021-10-25 at 11 44 49 PM" src="https://user-images.githubusercontent.com/90737587/138822890-79030ede-036d-497f-95f5-c1dd99fbb815.png">


Then, I created a table to display the various input parameters and output graphs for the Soliton Equation. Soliton is explained as a single wave and  the body  of  water  is  moving  as  one.  Another visual example is when  a  crowd  does  the  ”wave”  at  the sports stadium.  It was first discovered by John Scott Russell, who researched by visually watching the waves and came up with the basic properties of a Soliton.  Diederik Korteweg and Gustav de Vries created Soliton’s equation (KdV), and they also created a mathematical simulation of the KdV. 




 
## Soliton's Equation 

### Test 1: Input Code
```
# Build model
deep_stepper2 = keras.models.Sequential()
deep_stepper2.add(layers.Dense(10, input_dim=2, activation='elu'))
deep_stepper2.add(layers.Dense(10, activation='elu'))
deep_stepper2.add(layers.Dense(1, activation='linear'))

# Compile model
deep_stepper2.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_stepper2.fit(Xs_train, ys_train, epochs=10, batch_size=32,
            validation_data=(Xs_dev, ys_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
```            

### Test 1: Input Table 
<img width="900" alt="Screen Shot 2021-10-26 at 12 40 43 AM" src="https://user-images.githubusercontent.com/90737587/138830990-e9d845bf-82fc-4789-a241-8b72e2d242b6.png">

### Test 1: Output Graphs 
<img width="900" alt="Screen Shot 2021-10-26 at 1 08 50 AM" src="https://user-images.githubusercontent.com/90737587/138835435-85c26b62-6440-43f2-8327-9521b7356431.png">

### Test 2: Input Code
```
# Build model
deep_stepper2 = keras.models.Sequential()
deep_stepper2.add(layers.Dense(2, input_dim=6, activation='elu'))
deep_stepper2.add(layers.Dense(10, activation='elu'))
deep_stepper2.add(layers.Dense(1, activation='linear'))

# Compile model
deep_stepper2.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_stepper2.fit(Xs_train, ys_train, epochs=10, batch_size=32,
            validation_data=(Xs_dev, ys_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
```            

### Test 2: Input Table
<img width="900" alt="Screen Shot 2021-10-26 at 12 43 06 AM" src="https://user-images.githubusercontent.com/90737587/138831347-6acf083a-c4ec-4292-b4d5-bfa37dc0faa6.png">

### Test 2: Output Graphs
<img width="900" alt="Screen Shot 2021-10-26 at 12 43 41 AM" src="https://user-images.githubusercontent.com/90737587/138831451-fbd7075d-cd1f-47b5-9fbe-5b1648d8a91e.png">


