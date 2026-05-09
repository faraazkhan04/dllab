import tensorflow as tf

# Create tensors
a = tf.constant([1, 2, 3, 4], dtype=tf.float32)
b = tf.constant([5, 6, 7, 8], dtype=tf.float32)

# 1. Addition
add = tf.add(a, b)
print("Addition:", add.numpy())

# 2. Subtraction
sub = tf.subtract(b, a)
print("Subtraction:", sub.numpy())

# 3. Multiplication
mul = tf.multiply(a, b)
print("Multiplication:", mul.numpy())

# 4. Division
div = tf.divide(a, b)
print("Division:", div.numpy())

# 5. Square Operation
square = tf.square(a)
print("Square:", square.numpy())

# 6. Reshape Operation
reshape = tf.reshape(a, (4,))
print("Reshaped Tensor:")
print(reshape.numpy())


