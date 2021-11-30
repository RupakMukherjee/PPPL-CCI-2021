# Solving Partial Differential Equations using Machine Learning 

<br />

## Introduction
This project contains the use of machine learning utilizing an open-sourced software library, TensorFlow, to solve partial differential equations (PDEs). PDEs are equations used in physics and engineering applications, and the PDEs we will solve is the Advection-Diffusion equation, Burger’s equation, and Korteweg–De Vries (KdV) equation. First, we will go over what machine learning is, how TensorFlow can be utilized in machine learning, and how machine learning can be used to solve PDEs. Then, we will learn the history and real-world applications of these PDEs. Lastly, we will go over how to evaluate models and see how I trained the models to improve and produce more accurate, useful models.


| Name| Equation| Equation Graph|
|:---:|:--------:|:------:|
|Advection-Diffusion|<img width="150" src="https://user-images.githubusercontent.com/90737587/141264714-1cf3c558-669d-40dc-9635-1f680e8ade5e.png">|<img width="225" alt="Screen Shot 2021-10-31 at 9 01 22 PM" src="https://user-images.githubusercontent.com/90737587/139620001-ab139012-a904-4bce-8c1a-660ef648a118.png">|
|Burger|<img width="150"  src="https://user-images.githubusercontent.com/90737587/144005581-6fc42716-2252-47ba-b4f2-1e4fad1bd5de.png">|<img width="225" alt="Screen Shot 2021-11-03 at 1 11 57 AM" src="https://user-images.githubusercontent.com/90737587/140027116-32d19225-38f4-46f2-acf9-b82c0cb9e4db.png">|
|KdV|<img width="200"  src="https://user-images.githubusercontent.com/90737587/141266138-d94e8f4f-ede9-4257-8485-44ffa25c7275.png">|<img width="225" alt="Screen Shot 2021-10-31 at 8 30 29 PM" src="https://user-images.githubusercontent.com/90737587/139617944-881010bb-8643-42a4-947d-4a9a221482c7.png">|



## What is Machine Learning? <br />
Machine learning is the use of algorithms that can improve by evaluating past experiences and analyzing historical data. A traditional algorithm is a procedure for solving a well-defined task with some input and some given data to produce an output. 


<h3 align="center"> Traditional Algorithm </h3>
<p align="center">                    
<img src= "https://user-images.githubusercontent.com/90737587/141258601-a22f423f-b4fe-47ff-8604-84bca03b672b.png" width="90%" height="90%">
</p>


Let’s do a traditional algorithm example with the organization of a book collection. The task would be to organize the old books by genre such as action, mystery, historical fiction, horror… etc. Then, you would classify what defines each book genre and create a procedure for organizing the books. The output is the book collection organized by genre. 

<h3 align="center"> Traditional Algorithm </h3>
<p align="center">
<img src="https://user-images.githubusercontent.com/90737587/141258654-e01bdbb0-a2cd-4e7a-87c4-568642e7bc49.png" width="90%" height="90%">
</p>


In contrast, a machine learning algorithm is given a problem, some input with some shared data, and the output would be to produce a procedure. A machine learning algorithm learns patterns in data and can improve its procedure.

<h3 align="center"> Machine Learning Algorithm </h3>          
<p align="center">  
<img src="https://user-images.githubusercontent.com/90737587/141258702-750acb4a-9cf3-4499-9cf4-5f96142b6566.png" width="90%" height="90%">
</p>

Now, let’s try the same example we used for the traditional algorithm but this time with the machine learning algorithm. The task remains the same: to organize the books by genre such as action, mystery, historical fiction, horror… etc. Then, we should still input some rules of what defines each book genre. However, we do not create a procedure for organizing these books. Instead, we show the computer hundreds or thousands of examples of books and their genres. The computer will learn patterns on how to different books by genre on its own. In addition, over time, the computer will improve its procedure and identify and organize future books added to the book collection. 


<h3 align="center"> Machine Learning Algorithm </h3> 
<p align="center">
<img src="https://user-images.githubusercontent.com/90737587/141258799-dd6886a6-1be0-4c28-b92b-cd11a9bb343d.png" width="90%" height="90%">
</p>

The difference between a traditional algorithm and a machine learning algorithm is that a programmer would need to create a procedure for the traditional algorithm. Whereas, in machine learning, the computer would be responsible for creating a procedure. In addition, the computer can improve its procedure over time in machine learning, but a traditional algorithm cannot. How is this possible? Well, machine learning algorithms can improve because it utilizes a software library containing data that gives the algorithms the ability to improve.
<br />
<br />
## How TensorFlow is used in Machine Learning?
The software library we will use that gives us the ability for the machine learning algorithms to improve is TensorFlow. TensorFlow is an open-sourced software library that is utilized for training neural networks and models. TensorFlow is also known for solving numerical computations. Hence, we can use TensorFlow to solve PDEs. PDEs are used in physics and engineering applications such as simulating waves, heat flow, fluid dispersion, electrostatics over time, and more. The PDEs that we will use in the project are the Advection-Diffusion equation, Burger's equation, and KdV equation. But before we go more in-depth about what PDEs are and their real-world applications, let's go over the machine learning process first. 
<br />
<br />
## Machine Learning Process Flowchart

The flowchart below will explain the five steps of the general machine learning process. We want to get a great understanding of the basics before we go into detail about how the machine learning process is utilized to solve PDEs. First, start at step 1, then step 2 until you end at step 5.  

<img src="https://user-images.githubusercontent.com/90737587/141751967-9b24e444-f079-4771-95a4-5eacdcc652ee.png" width="100%" height="100%">

For this project, I utilized this [Advection-Diffusion code](https://github.com/dynamicslab/deep-learning-fluids/blob/main/notebooks/advection_diffusion_DL.ipynb) that also uses the same general machine learning process shown above. I used the same code for the Burger’s and KdV’s code except with some slight modifications. In the flowchart down below, I will explain how we utilized the machine learning process in regard to solving PDEs. Keep in mind that I will be solely focusing on the machine learning process itself. Later, I will go into further detail about what Tests 1 & 2 are and how to evaluate and improve the models. With that said, for steps 1-3, I will explain what the creator of the Advection-Diffusion code did for data collection, preparing the data, and building the model. Then, for steps 3-5, I will explain what I have done to train the model, test the model, and improve the model to get better results. 
<br />
<br />
## How was the Machine learning process used in this project?

<img src= "https://user-images.githubusercontent.com/90737587/142388491-a1046e69-e6cd-4774-ac44-a5fce6405dd2.png" >

*Note: Though Test 2 did not need improvement, I still tuned parameters of model to verify the accuracy of the model.*  
<br />
## What is a Partial Differential Equation (PDE) and its relation to Machine Learning?

Like mentioned prior, PDEs are mathematical equations used in physics and engineering applications such as simulating waves, heat flow, fluid dispersion, electrostatics over time, and more. PDEs contain partial derivatives of functions that depend on several variables. PDEs are not generally solvable analytically, only a few can be solved with exact solutions. For the PDEs that we have yet to produce exact solutions for, researchers have developed numerical solution techniques on the computer to create approximated solutions and simulations over the years. This project will demonstrate how to use machine learning and TensorFlow to produce approximated solutions for these PDEs. The PDEs we will learn about are the Advection-Diffusion equation, Burger’s equation, and the KdV equation.
<br />
<br />
## What is the Advection-Diffusion equation?
The Advection-Diffusion equation describes a large amount of substance, usually fluid, that is spread out over a period time. Figure 1 shows oil being dumped into a river and spreading out. The Advection-Diffusion equation explains how the concentration of oil in the water changes over time.

| Equation| Equation Graph| Real-world Example |
|:--------:|:------------:|:------------------:|
|<img width="150" src="https://user-images.githubusercontent.com/90737587/141264714-1cf3c558-669d-40dc-9635-1f680e8ade5e.png">|<img width="225" alt="Screen Shot 2021-10-31 at 9 01 22 PM" src="https://user-images.githubusercontent.com/90737587/139620001-ab139012-a904-4bce-8c1a-660ef648a118.png">|<img width="250" src="https://user-images.githubusercontent.com/90737587/142381663-0634b474-ad8e-4c73-844d-78c9d220adde.jpg">  <br /> (Figure 1: Oil spill in river)|

### Defining the variables:
t = temporal coordinate *(a specific time)*     <br />
u(x,t) = speed of fluid *(speed dependent on x and t)*  
D = diffusion coefficient *(amount of a particle substance that diffuses across an unit of area)*  
v = advection coefficient *(velocity or speed of the flow)* <br />
<br />

### The History of the Advection-Diffusion equation:
The Advection-Diffusion equation is of fluid dynamics which is the study of fluid motion under the action of physics forces. The term "advection" means the transport of a large quantity of fluid. The term "diffusion" means the intermingling of substances by the natural movement of their particles. The Advection-Diffusion equation is derived from the continuity equation. The continuity equation describes the transport of a quantity and is of one of physics fundamentals laws, the conservation law. The conservation law states that energy can neither be created nor destroyed, but it can transform one form into another. 
<br />
<br />

## What is the Burger's equation?
Burger’s equation resembles a shock wave and is known for modeling turbulence. Turbulence is fluid motion caused by dramatic, hectic changes in pressure and flow velocity. If you have been an airplane, you may be familiar with turbulence as the cause of an uneasy and not so smooth ride for a short period of time. Figure 2 shows a drawing displaying how wind turbulence may interact with an airplane. Burger's equation illustrates the theory of turbulence.

| Equation| Equation Graph| Real-world Example |
|:--------:|:------------:|:------------------:|
|<img width="150" src="https://user-images.githubusercontent.com/90737587/144005581-6fc42716-2252-47ba-b4f2-1e4fad1bd5de.png">|<img width="240" src="https://user-images.githubusercontent.com/90737587/140027116-32d19225-38f4-46f2-acf9-b82c0cb9e4db.png">|<img width="240" src="https://user-images.githubusercontent.com/90737587/142951722-8d60f67e-9187-4981-b8b4-79af1aff019e.png"> <br /> (Figure 2: Wind turbulence interacting with an airplane )|

### Defining the variables:  
x = spatial coordinate *(a specific location)*   <br />
t = temporal coordinate *(a specific time)*    <br />
u(x,t) = speed of fluid *(speed dependent on x and t)*     
v = advection coefficient *(velocity or speed of the flow)*    <br />
<br />

### The History of the Burger's equation: 
The Burger’s equation or Bateman-Burger’s equation was created by Harry Bateman in 1915 and then later studied by Johanne Martinus Burgers in 1948. Burger’s equation is obtained by combining a nonlinear wave motion with a linear diffusion and is the simplest nonlinear model equation for diffusive waves in fluid dynamics. Burger’s equation is derived from the Naiver Strokes equation. Naive Strokes equations are the fundamental equations of viscous fluid dynamics. Similar to the Advection-Diffusion equation, Burger’s equation is a continuity equation and from the conservation law.
<br />
<br />

## What is the KdV's equation?

The Korteweg–De Vries (KdV) equation explains the shape and movement of solitons. A soliton, also known as a solitary wave, is a wave packet that maintains its shape at constant velocity. Tsunamis behave similarly to solitons. For instance, Figure 3 illustrates the movement of the earth’s crust underwater, an earthquake, causing a long, large wave packet to travel a long distance while maintaining its wave packet speed and shape, also known as a tsunami. Meaning, the KdV equation can describe the shape and movement of Tsunamis. 

| Equation| Equation Graph| Real-world Example |
|:--------:|:------------:|:------------------:|
|<img width="150" src="https://user-images.githubusercontent.com/90737587/141266138-d94e8f4f-ede9-4257-8485-44ffa25c7275.png">|<img width="225" src="https://user-images.githubusercontent.com/90737587/139617944-881010bb-8643-42a4-947d-4a9a221482c7.png"> |<img width="260" src="https://user-images.githubusercontent.com/90737587/144011938-10346ad2-1e75-458c-a758-f1642e5e23b9.png">  <br /> (Figure 3: A digram of a tsunami)|

### Defining the variables: 
x = spatial coordinate (a specific location)  <br />
t = temporal coordinate (a specific time)   <br />
u(x,t) = speed of fluid (speed dependent on x and t)  <br />
D = diffusion coefficient (amount of a particle substance that diffuses across an unit of area)  <br />
v = advection coefficient (velocity or speed of the flow)  <br />
<br />

### The History of the KdV's equation:
The history of the KdV equation started with John Scott Russell observing a solitary wave, also known as a soliton, in a canal in 1834. Following this, Russell came up with the basic properties of a soliton. In 1870, Lord Rayleigh and Joseph Boussinesq did further theoretical investigations. Then in 1895, Korteweg and De Vries created the KdV equation, a mathematical model of soliton waves in shallow waters. In 1965, Norman Zabusky and Martin David Kruskal at Princeton Physics Plasma laboratory created computer simulations to study the KdV equation. They discovered the solitons could keep their shape after passing through each other. In 1967, Gardner, Greene, Kruskal, and Miura developed an analytic solution to the KdV equation utilizing the inverse scattering transform. In other words, the KdV equation is of the few non-linear partial differential equations that can be solved with exact solutions. Like Burger’s equation and the Advection-Diffusion equation, the KdV equation is derived from the conservation law. 

# Test 1
## Advection-Diffusion Equation

### Test 1: Input code 
```
# Build model
deep_approx = keras.models.Sequential()
deep_approx.add(layers.Dense(2, input_dim=2, activation='elu'))
deep_approx.add(layers.Dense(10, activation='elu'))
deep_approx.add(layers.Dense(1, activation='linear'))

# Compile model
deep_approx.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_approx.fit(X_train, y_train,
            epochs=10, batch_size=32,
            validation_data=(X_dev, y_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
```

### Test 1: Input Table 

| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Input Dimension |Epochs|
|:---:|:------------: | :-----------: |:------------:|:--------------:|:---:|
|[A1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA1.py)|2|10|1|2|10|
|[A2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA2.py)|5|25|3|2|30|
|[A3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA3.py)|50|2|1|2|30|
|[A4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA4.py)|100|100|1|2|30|
|[A5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA5.py)|1000|1000|10|2|30|

### Test 1: Output Graphs 

| Test| Validation Loss Chart | Error Chart |
|:---:|:--------------------: | :-----------:|
|[A1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA1.py)|<img width="225" alt="Screen Shot 2021-11-01 at 2 55 05 PM" src="https://user-images.githubusercontent.com/90737587/139747239-9f22f746-082e-4a5a-ac22-bc0f59479bc0.png">|<img width="375" alt="Screen Shot 2021-11-01 at 2 07 12 PM" src="https://user-images.githubusercontent.com/90737587/139742276-8f0f9c77-617c-4510-9656-8bac4d6eff84.png">|
|[A2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA2.py)|<img width="225" alt="Screen Shot 2021-11-01 at 2 26 46 PM" src="https://user-images.githubusercontent.com/90737587/139744302-87f30127-f43a-4081-973a-07999db88091.png">|<img width="375" alt="Screen Shot 2021-11-01 at 2 27 03 PM" src="https://user-images.githubusercontent.com/90737587/139744324-0bfc54ef-7cbe-4c29-a887-ff9a48e43133.png">|
|[A3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA3.py)|<img width="225" alt="A3VL" src="https://user-images.githubusercontent.com/90737587/139741080-58744d79-e820-4d3b-9f76-c4130a74463c.png">|<img width="375" alt="A3EC" src="https://user-images.githubusercontent.com/90737587/139741092-8ad564b5-845f-438f-80e0-3a3fa780edff.png">|
|[A4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA4.py)|<img width="225" alt="A4VL" src="https://user-images.githubusercontent.com/90737587/139741116-4b148900-b5b9-4d6f-a5e3-3a3c5bb5bfb8.png">|<img width="375" alt="A4EC" src="https://user-images.githubusercontent.com/90737587/139741133-03c6209a-77af-4fb8-908c-6a19d0ce5ed3.png">|
|[A5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA5.py)|<img width="225" alt="A5VL" src="https://user-images.githubusercontent.com/90737587/139741162-91d00fe1-9b77-418f-94b9-e4cd9a8b75d8.png">|<img width="375" alt="A5EC" src="https://user-images.githubusercontent.com/90737587/139741178-994d891d-9b68-4524-8aae-652d53983199.png">|

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
            
```
[//]: <> (Changed names of Test cases. Old Name: A, New name: B1)

### Test 1: Input Table 

| Test| Dense Layer 1| Dense Layer 2| Dense Layer 3| Epochs|
|:---:|:------------: | :-----------: |:------------:|:---:|
|[B1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB1.py)|10 |10 |1 |10|
|[B2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB2.py)|2 |10 |1 |30|
|[B3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB3.py)|4 |4 |5 |30|
|[B4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB4.py)|15 |15 |2 |30|
|[B5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB5.py)|50 |50 |1 |30|

### Test 1: Output Graphs 

| Test| Validaiton Loss Chart| Error Chart|
|:---:|:------------:|:-----------:|
|[B1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB1.py)|<img width="225" alt="TestA-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619306-480ea9e0-f654-4864-bda3-ae5a0f53c418.png">|<img width="375" alt="TestA-Error (1)" src="https://user-images.githubusercontent.com/90737587/139619329-5b7955f1-579b-4ee3-a147-bd3197126564.png">|
|[B2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB2.py)|<img width="225" alt="TestB-ValLoss (4)" src="https://user-images.githubusercontent.com/90737587/139619412-95f597ab-cbfc-4063-8eaa-30f4491a43e8.png">|<img width="375" alt="TestB-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139619425-680dee6c-396c-4ae8-a8cd-bcb17c41b372.png">|
|[B3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB3.py)|<img width="225" alt="TestC-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619474-6810223d-fcfc-436e-a9f0-3b4386f0b234.png">|<img width="375" alt="TestC-Error (1)" src="https://user-images.githubusercontent.com/90737587/139619498-d43d8a6b-786e-4d54-bcdf-3486bd74d326.png">|
|[B4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB4.py)|<img width="225" alt="TestD-ValLoss (6)" src="https://user-images.githubusercontent.com/90737587/139619635-e958cf35-bacd-4a92-a7cf-759393356139.png">|<img width="375" alt="TestD-Error (5)" src="https://user-images.githubusercontent.com/90737587/139619653-7c44d6e6-4caa-433b-91ea-e701b2f2ccad.png">|
|[B5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB5.py)|<img width="225" alt="TestE-VaLoss (2)" src="https://user-images.githubusercontent.com/90737587/139619684-dc647f49-c17e-4137-820a-4be6be4dfa66.png">|<img width="375" alt="TestE-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139619696-ffe55108-72fa-454e-ae4b-9ecc3a61a288.png">|

## KdV's Equation 

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

[//]: <> (Changed names of Test cases. Old Name: 1, New name: K1)

### Test 1: Input Table 

| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Epochs|
|:---:|:-------------:| :-----------: |:------------:|:----:|
|[K1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK1.py)|10|10|1|10|
|[K2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK2.py)|5|5|1|30|
|[K3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK3.py)|50|50|1|30|
|[K4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK4.py)|150|150|1|30|
|[K5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK5.py)|500|500|1|100|

### Test 1: Output Graphs 

| Test| Validation Loss Chart | Error Chart |
|:---:|:--------------------: | :-----------:|
|[K1](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK1.py)|<img width="225" alt="Test1-ValLoss (1)" src="https://user-images.githubusercontent.com/90737587/139624198-16c6e49e-6926-4417-8c17-301a1536c738.png">|<img width="375" alt="Test1-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139624213-41c3c1ac-87dc-4698-8266-317d3d38a0ac.png">|
|[K2](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK2.py)|<img width="225" alt="Test2-ValLoss (1)" src="https://user-images.githubusercontent.com/90737587/139624233-8fdecb8c-bb72-4220-ac18-5611e7fb219c.png">|<img width="375" alt="Test2-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139624246-9ff385be-e0f5-44c1-9de2-f1f200a8aae2.png">|
|[K3](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK3.py)|<img width="225" alt="Test3-ValLoss (1)" src="https://user-images.githubusercontent.com/90737587/139624281-08b27a62-20ab-4aef-9fa5-5aa32823205c.png">|<img width="375" alt="Test3-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139624300-bfdd214b-6ea2-4cfb-99be-a7fdf9406c33.png">|
|[K4](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK4.py)|<img width="225" alt="Test4-ValLoss (3)" src="https://user-images.githubusercontent.com/90737587/139624342-113f98ce-975d-4514-85fb-1ea3aae07ffe.png">|<img width="375" alt="Test4-Errors (1)" src="https://user-images.githubusercontent.com/90737587/139624355-3b8d94b9-5fe4-438a-9811-a32b04ca4c44.png">|
|[K5](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK5.py)|<img width="225" alt="Test5-ValLoss (2)" src="https://user-images.githubusercontent.com/90737587/139624404-4eaa9878-f9a2-43d3-b5d7-185af74525e3.png">|<img width="375" alt="Test5-Errors (2)" src="https://user-images.githubusercontent.com/90737587/139624426-ebf490e7-6a90-413c-b1c2-950596c9af82.png">|

# Test 2
## Advection-Diffusion Equation

### Test 2: Input Code
```
# Build model
deep_stepper2 = keras.models.Sequential()
deep_stepper2.add(layers.Dense(2, input_dim=3, activation='elu'))
deep_stepper2.add(layers.Dense(10, activation='elu'))
deep_stepper2.add(layers.Dense(1, activation='linear'))

# Compile model
deep_stepper2.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_stepper2.fit(Xs_train, ys_train, epochs=3, batch_size=32,
            validation_data=(Xs_dev, ys_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
            
```
### Test 2: Input Table 

| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Input Dimension |Epochs|
|:---:|:------------: | :-----------: |:------------:|:--------------:|:---:|
|[A6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA6.py)|2|10|1|3|1|
|[A7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA7.py)|50|70|1|2|10|
|[A8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA8.py)|100|100|1|2|30|

### Test 2: Output Graphs 

| Test| Graph 1| Graph 2| Graph 3| Graph 4|
|:---:|:------:|:------:|:------:|:------:|
|[A6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA6.py)|<img width="300" alt="A6E1" src="https://user-images.githubusercontent.com/90737587/140024192-7196a68a-18f5-4190-9b68-5a4f742f58d0.png">|<img width="300" alt="A6fu" src="https://user-images.githubusercontent.com/90737587/140024224-3dd44f8e-77ab-4745-8100-109e2c48157b.png">|<img width="300" alt="A6ET" src="https://user-images.githubusercontent.com/90737587/140024242-9e504113-4612-4c2a-92e8-35fa357e4290.png">|<img width="300" alt="A6u" src="https://user-images.githubusercontent.com/90737587/140024573-3617c695-7b03-41ee-a34a-cb05edd99e60.png">|
|[A7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA7.py)|<img width="300" alt="A7E1" src="https://user-images.githubusercontent.com/90737587/140024348-5af83f54-df94-40a9-b313-04af79951a02.png">|<img width="300" alt="A7Fu" src="https://user-images.githubusercontent.com/90737587/140024379-da1b7e05-6854-4f09-a475-e9a25703837e.png">|<img width="300" alt="A7ET" src="https://user-images.githubusercontent.com/90737587/140024410-145239dc-7dc9-4816-a068-164c83f18ced.png">|<img width="300" alt="A7u" src="https://user-images.githubusercontent.com/90737587/140024544-2773c4eb-6882-4dd6-8091-e7ad6646e34c.png">|
|[A8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testA8.py)|<img width="300" alt="A8E1" src="https://user-images.githubusercontent.com/90737587/140026659-c9db5b16-e527-466b-a938-f90d548943cd.png">|<img width="300" alt="A8Fu" src="https://user-images.githubusercontent.com/90737587/140026681-f11d3f2a-ef2d-4d99-96ad-3ae1151a5232.png">|<img width="300" alt="A8ET" src="https://user-images.githubusercontent.com/90737587/140026702-b5ad76d3-9339-4c4f-bdbf-1c3b0e1da1d3.png">|<img width="300" alt="A8u" src="https://user-images.githubusercontent.com/90737587/140026708-d87c17d9-5519-4f81-af8d-5aeaa120034b.png">|



## Burger's Equation

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

| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Epochs|
|:---:|:-------------:|:-------------:|:------------:|:----:|
|[B6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB6.py) |2|10|1|3|
|[B7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB7.py) |10|15|1|10|
|[B8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB8.py) |50|80|5|30|

### Test 2: Output Graphs

| Test| Graph 1| Graph 2| Graph 3| Graph 4|
|:---:|:------:|:------:|:------:|:------:|
|[B6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB6.py)  |<img width="300" alt="TestF-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139620520-bdf3bf2b-2b90-44e1-85af-4339d254cbc3.png">|<img width="300" alt="TestFu(x,t) (1)" src="https://user-images.githubusercontent.com/90737587/139620537-30f6e297-8219-4b58-9c7d-b23beaf2cda2.png">|<img width="300" alt="TestG-ErrorT (1)" src="https://user-images.githubusercontent.com/90737587/139620547-cf8adece-16e8-4b6e-9e96-d57b75eac463.png">|<img width="300" alt="TestF-u(x,Tend) (1)" src="https://user-images.githubusercontent.com/90737587/139620559-0d169b96-4952-405b-a66d-722c805af42f.png">|
|[B7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB7.py) | <img width="300" alt="TestG-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139623766-2467b415-e4b9-480c-b7e1-6c2e74b59bdd.png">|<img width="300" alt="TestG-u(x,t) (1)" src="https://user-images.githubusercontent.com/90737587/139623777-a8f78f57-f409-4f78-8985-1aabbfcee912.png">|<img width="300" alt="TestG-ErrorT (2)" src="https://user-images.githubusercontent.com/90737587/139623789-8ff717d4-b8b2-4f82-94ca-1bac52131eeb.png">|<img width="300" alt="TestG-u(x, tend) (1)" src="https://user-images.githubusercontent.com/90737587/139623814-0c04324b-de64-46f2-943c-ea8e3bc3baf2.png">|
|[B8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testB8.py)|<img width="300" alt="TestH-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139623861-c9e5f383-e65b-48a0-995f-74caab44ad66.png">|<img width="300" alt="TestH-u(x,t) (1)" src="https://user-images.githubusercontent.com/90737587/139623876-0ec93ccc-7324-47ac-9148-5501b3094427.png">|<img width="300" alt="TestH-ErrorT (1)" src="https://user-images.githubusercontent.com/90737587/139623889-1c849db1-cffa-48c3-a7a0-6b4573c17290.png">|<img width="300" alt="TestH-u(x, tend) (2)" src="https://user-images.githubusercontent.com/90737587/139623897-d809bf26-5898-49ca-8d12-d14740c4fd91.png">|



## KdV's Equation 

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

| Test| Dense Layer 1 | Dense Layer 2 |Dense Layer 3 |Epochs|Input Dimension|
|:---:|:------------: | :-----------: |:------------:|:---:|:--------------:|
|[K6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK6.py)|2|10|1|10|6|
|[K7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK7.py)|50|50|1|30|6|
|[K8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK8.py)|100|80|1|50|2|


### Test 2: Output Graphs

| Test| Graph 1| Graph 2| Graph 3| Graph 4|
|:---:|:------:|:------:|:------:|:------:|
|[K6](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK6.py)|<img width="300" alt="Test6-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139624811-a40e83e2-ea3a-4065-bfd8-a00fe5281273.png">|<img width="300" alt="Test6-u(x,t) (2)" src="https://user-images.githubusercontent.com/90737587/139624825-f03e014f-c797-4586-878f-99ca86f624e8.png">|<img width="300" alt="Test6-ErrorT (1)" src="https://user-images.githubusercontent.com/90737587/139624843-bba92f16-9950-4997-9bd9-c92b8be7de14.png">|<img width="300" alt="Test6-u(x,tend)" src="https://user-images.githubusercontent.com/90737587/139624854-dad74923-4298-450d-b69b-dd7873bc5160.png">|
|[K7](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK7.py)|<img width="300" alt="Test7-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139624870-890cff00-7891-4b3d-8602-a40e0f813877.png">|<img width="300" alt="Test7-u(x,t) (1)" src="https://user-images.githubusercontent.com/90737587/139624894-265ee0bd-b6fc-4708-ae84-dc393f5f16b8.png">|<img width="300" alt="Test7-ErrorT (1)" src="https://user-images.githubusercontent.com/90737587/139624905-29921347-a660-4bfc-9ef2-5e359cba12ec.png">|<img width="300" alt="Test7-u(x,tend) (2)" src="https://user-images.githubusercontent.com/90737587/139624918-037d671b-4553-49eb-bc03-49b9089b1ad0.png">|
|[K8](https://github.com/RupakMukherjee/PPPL-CCI-2021/blob/main/testK8.py)|<img width="300" alt="Test8-Error1 (1)" src="https://user-images.githubusercontent.com/90737587/139624929-810d0381-f81a-4a9f-941b-76fda8410024.png">|<img width="300" alt="Test8-u(x,t) (1)" src="https://user-images.githubusercontent.com/90737587/139624940-1b745f77-adf9-4504-bfa2-7f49bc326a27.png">|<img width="300" alt="Test8-ErrorT (1)" src="https://user-images.githubusercontent.com/90737587/139624946-e8389c11-8979-472d-80a3-a9c77c26cbcd.png">|<img width="300" alt="Test8-u(x,tend)" src="https://user-images.githubusercontent.com/90737587/139624976-e6d3e499-8263-4c71-823d-a9c1a29a93fb.png">|

