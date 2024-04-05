import numpy as np
import torch
import matplotlib.pyplot as plt


model = torch.load('neural-network-1.pth')
# creating s
step = 50
y = np.linspace(0, 1.5, step)
w = np.linspace(-np.pi, np.pi, step)

Y, W = np.meshgrid(y, w)
s = np.zeros((step**2, 8))

s[:, 1] = Y.flatten()
s[:, 4] = W.flatten()

# getting the max values (optimal policy)
values = model(torch.tensor(s, dtype=torch.float32, requires_grad=False))
max_values, argmax_values = values.max(1)

max_Q = max_values.detach().numpy().reshape((step, step))
argmax_Q = argmax_values.detach().numpy().reshape((step, step))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, W, max_Q, cmap='viridis', edgecolor='none')

ax.set_title('Max Q Values for varying Heights and Angles')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('Max Q')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, W, argmax_Q, cmap='viridis', edgecolor='none')

ax.set_title('Optimal Q actions for varying Heights and Angles')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('Action')
ax.set_zticks([0, 1, 2, 3])
ax.set_zticklabels(['Nothing', 'Left.', 'Main.', 'Right'])

plt.show()
