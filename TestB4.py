#Imports are deriving data from another file
import tensorflow as tf
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

#Variables
x_left = 0
x_right = 6.28
t_end = 1

D = 0.02
v = 0.1
dx = 0.01
dt = 0.001

#what is cfl stand for?
cfl = D*dt/dx**2
print('CFL = %.4f'%(cfl))
x_steps = int((x_right - x_left)/dx)
t_steps = int(t_end/dt)

#output will populate

#np.linespace function,array, numpy=np (start, stop, num=50, endpoint=true...)
xx = np.linspace(x_left, x_right, x_steps)
tt = np.linspace(0, t_end, t_steps)
uu = np.zeros((t_steps, x_steps))

# ICs , def function with parameters that returns a single value
#mu=mean, sig=sigma,
def gaussian(x, mu, sig, shift):
    return shift + np.sin(x)

ic_mean = 0
ic_std = 0.2
ic_shift = 0
uu[0, :] = gaussian(xx, ic_mean, ic_std, ic_shift)

# Euler Stepper
def adv_diff_euler_step(ul, uc, ur, D, v, dx, dt):
    return uc + D * dt/dx**2 * (ul - 2 * uc + ur) - uc * dt/(2*dx) * (ur - ul)

# Solve, arrays, loops
#enumerate, (index, value)
inputs = []
outputs = []
for ti, t in enumerate(tt[:-1]):
    for xi, x in enumerate(xx[1:-1]):
        uu[ti+1, xi] = adv_diff_euler_step(uu[ti, xi-1], uu[ti, xi], uu[ti, xi+1], D, v, dx, dt)

        # zero flux BC
        uu[ti+1, 0] = uu[ti+1, -1]
        uu[ti+1, 1] = uu[ti+1, -2]

        # Save data, continuting array, plotting
        inputs.append([t, x])
        outputs.append(uu[ti+1, xi])
        
fig = plt.figure()
plt.plot(xx, uu[0, :], lw=2)
plt.plot(xx, uu[-1, :], lw=2)
plt.xlabel('$x$')
plt.ylabel('$u(x, t)$')
plt.legend(['$t_0$', '$t_{end}$'])


plt.show()
#exit()

#neural network

test_ratio = 0.25
dev_ratio = 0.2

# Prepare data
inputs_array = np.asarray(inputs)
outputs_array = np.asarray(outputs)

# Split into train-dev-test sets
#training (from sklearn.model_selection)
X_train, X_test, y_train, y_test = train_test_split(inputs_array, outputs_array, test_size=test_ratio, shuffle=False)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_ratio, shuffle=False)

# Build model, neutral network
#squential model, each layer has one input tensor, one output tensor (3)
#elu has negative values which allows for answer to be closer to zero and more accurate
#1D, 10 units, linear- turns the neural network into one layer
#total param: it # of # that model will adjust in order to learn
deep_approx = keras.models.Sequential()
deep_approx.add(layers.Dense(15, input_dim=2, activation='elu'))
deep_approx.add(layers.Dense(15, activation='elu'))
deep_approx.add(layers.Dense(1, activation='linear'))

# Compile model, will make guess using loss & optimizer, slowly get closer to correct formula. model guesses, loss will show, then optmizer will generate another guess
deep_approx.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_approx.fit(X_train, y_train,
            epochs=30, batch_size=32,
            validation_data=(X_dev, y_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))
            
#displays squential model
deep_approx.summary()

# history.history contains loss information

idx0 = 1
plt.figure()
plt.plot(history.history['loss'][idx0:], '.-', lw=2)
plt.plot(history.history['val_loss'][idx0:], '.-', lw=2)
plt.xlabel('epochs')
plt.ylabel('Validation loss')
plt.legend(['training loss', 'validation loss'])
plt.show()

import seaborn as sns
c = sns.color_palette()

nplots = 11
rmin = 0
rmax = 1
idxes = np.arange(int(rmin*len(tt)), int(rmax*len(tt)), int((rmax-rmin)*len(tt)/nplots))
e_mean = []
tt_mean = []

#enumerate allows you to loop over an iterable object and keep track of how many iterations have occurred.
fig = plt.figure(figsize=(12, 4))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for idx, i in enumerate(idxes):
    data_in = np.array([ [tt[i], x] for x in xx])
    u_approx = deep_approx.predict(data_in)
    ax0.plot(xx, u_approx, lw=2, color=c[idx%len(c)])
    ax0.plot(xx, uu[i, :], lw=2, linestyle='--')
    tt_mean.append(tt[i])
    e_mean.append( np.mean((u_approx[:, 0] - uu[i, :])**2) )

ax1.plot(tt_mean, e_mean, '.-', lw=2, color=c[0], markersize=10)
ax1.plot([(1-test_ratio)*t_end]*2, [min(e_mean), max(e_mean)], ':', color=c[1])
ax1.legend(['RMSE', 'Train/dev time horizon'])

ax0.set_xlabel('$x$')
ax0.set_ylabel('$u(x, t)$')
# ax0.legend(['$t^*_{end}$'])
ax1.set_ylabel('Error')

fig.tight_layout()
plt.show()
exit()
#below is test 2
inputs = []
outputs = []
for ti, t in enumerate(tt[:-1]):
    for xi, x in enumerate(xx[1:-1]):
        uu[ti+1, xi] = adv_diff_euler_step(uu[ti, xi-1], uu[ti, xi], uu[ti, xi+1], D, v, dx, dt)
        uu[ti+1, 0] = uu[ti+1, 1]
        uu[ti+1, -1] = uu[ti+1, -2]

        # Collect data
        inputs.append([uu[ti, xi-1], uu[ti, xi], uu[ti, xi+1]])
        outputs.append(uu[ti+1, xi])

inputs_array = np.asarray(inputs)
outputs_array = np.asarray(outputs)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(inputs_array, outputs_array, test_size=test_ratio, shuffle=False)
Xs_train, Xs_dev, ys_train, ys_dev = train_test_split(Xs_train, ys_train, test_size=dev_ratio, shuffle=True)

## linear regression of stepper

# Build model
deep_stepper = keras.models.Sequential()
deep_stepper.add(layers.Dense(1, input_dim=3, activation='linear'))

# Compile model
deep_stepper.compile(loss='mse', optimizer='adam')

# Fit!
history = deep_stepper.fit(Xs_train, ys_train, epochs=3, batch_size=32,
            validation_data=(Xs_dev, ys_dev),
            callbacks=keras.callbacks.EarlyStopping(patience=5))

## Integrate with the neural network
# WARNING (and lesson): this will take too long! (around 10 minutes)
# Can accelerate by vectorizing input_stencil
    
from tqdm import tqdm

uu_deep = np.zeros(uu.shape)
uu_deep[0, :] = uu[0, :]

for ti in tqdm(range(len(tt[:-1]))):
    for xi, x in enumerate(xx[1:-1]):
        input_stencil = np.array([[uu[ti, xi-1], uu[ti, xi], uu[ti, xi+1]]])
        uu_deep[ti+1, xi] = deep_stepper( input_stencil )[0][0].numpy()
    uu_deep[ti+1, 0] = uu_deep[ti+1, 1]
    uu_deep[ti+1, -1] = uu_deep[ti+1, -2]
    
    fig = plt.figure()
plt.plot(xx, uu_deep[-1, :], lw=2)
plt.plot(xx, uu[-1, :], '--', lw=2)
plt.xlabel('$x$')
plt.ylabel('$u(x, t_{end})$')
plt.legend(['Deep stepper', 'Euler stepper'])

idx_list = [0, int(len(tt)/4), int(len(tt)/2), int(len(tt))-1]
leg = []
fig = plt.figure()
for idx in idx_list:
    plt.plot(xx, (uu_deep[idx, :] - uu[idx, :])**2, lw=2)
    leg.append('$t=%.2f s$'%(tt[idx]))
plt.xlabel('$x$')
plt.ylabel('$Error$')
plt.legend(leg)
plt.show()

# If the time stepper is linear. Does it learn the right coefficients?

# Euler step: uc + D * dt/dx**2 * (ul - 2 * uc + ur) + v * dt/(2*dx) * (ur - ul)

weights = deep_stepper.get_weights()[0]
bias = deep_stepper.get_weights()[1]

print("actual coefficient of u_left is %.5f and the fit is %.5f"%(D*dt/dx**2 - v*dt/(2*dx), weights[0]))
print("actual coefficient of u_center is %.5f and the fit is %.5f"%(-2*D*dt/dx**2 + 1, weights[1]) )
print("actual coefficient of u_right is %.5f and the fit is %.5f"%(D*dt/dx**2 + v*dt/(2*dx), weights[2]))
print(bias)

## In general, you're not guaranteed to get the same solution.

## Nonlinear regression of stepper

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

## Integrate with the neural network
# WARNING (and lesson): this will take too long! (around 10 minutes)
# Can accelerate by vectorizing input_stencil

uu_deep2 = np.zeros(uu.shape)
uu_deep2[0, :] = uu[0, :]

for ti, t in enumerate(tt[:-1]):
    # print(ti/len(tt))
    for xi, x in enumerate(xx[1:-1]):
        input_stencil = np.array([[uu[ti, xi-1], uu[ti, xi], uu[ti, xi+1]]])
        uu_deep2[ti+1, xi] = deep_stepper2( input_stencil )[0][0].numpy()
    uu_deep2[ti+1, 0] = uu_deep2[ti+1, 1]
    uu_deep2[ti+1, -1] = uu_deep2[ti+1, -2]

fig = plt.figure()
plt.plot(xx, uu_deep[-1, :], lw=2)
plt.plot(xx, uu_deep2[-1, :], lw=2)
plt.plot(xx, uu[-1, :],  '--', lw=2)
plt.xlabel('$x$')
plt.ylabel('$u(x, t)$')
plt.legend(['Deep stepper', 'Euler stepper'])


idx_list = [0, int(len(tt)/4), int(len(tt)/2), int(len(tt))-1]
leg = []
fig = plt.figure()
for idx in idx_list:
    plt.plot(xx, (uu_deep[idx, :] - uu[idx, :])**2, lw=2)
    leg.append(str(tt[idx]))
plt.xlabel('$x$')
plt.ylabel('$Error$')
plt.legend(leg)
plt.show


