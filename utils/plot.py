import matplotlib.pyplot as plt

train_losses = [7.840795, 7.776605, 7.773864, 7.760384, 7.751472, 7.763744, 7.743400, 7.745626, 7.759824, 7.729642, 7.754485, 7.758963, 7.742980, 7.748306, 7.744513, 7.748112, 7.736103, 7.744472]
epochs = range(0, len(train_losses))

plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
