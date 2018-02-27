# OpenAI XOR

Warm-up exercise from OpenAI's [Requests for Research](https://blog.openai.com/requests-for-research-2/).

### Task
Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end. Test the two approaches below:

- Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
- Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

### Solution
Using the helper functions defined in `utils.py`:

```python
from utils import build_lstm, compute_accuracy
from utils import labels, random_binary_data

n_train = 100000
n_test  = 1000
epochs  = 20

# Generate Data
x_constant = random_binary_data(n_train, min_len=50, max_len=50)
x_varied   = random_binary_data(n_train, min_len=1,  max_len=50)
x_test     = random_binary_data(n_test,  min_len=50, max_len=50)

y_constant = labels(x_constant)
y_varied   = labels(x_varied)
y_test     = labels(x_test)

# Build the models
n_units = 10
model_constant = build_lstm(n_units)
model_varied   = build_lstm(n_units)

# Train and print results
model_constant.fit(x_constant, y_constant, epochs=epochs)
print("Accuracy: %f" % compute_accuracy(model_constant, x_test, y_test))
# Accuracy ~ 40%

model_varied.fit(x_varied, y_varied, epochs=epochs)
print("Accuracy: %f" % compute_accuracy(model_varied, x_test, y_test))
# Accuracy > 90%
```
