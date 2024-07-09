
# Step 3: Load the File
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import tensorflow as tf

# Verify TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the dataset
df = pd.read_csv('./data.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['text', 'target']]

# Map target to three classes: negative, neutral, positive
df['target'] = df['target'].map({0: 0, 4: 1})

# Sample 66,666 instances from each class to get a total of 200,000 samples
neg_df = df[df['target'] == 0].sample(n=66666, random_state=42)
neu_df = df[df['target'] == 1].sample(n=66666, random_state=42)
# Combine the sampled data
df_sampled = pd.concat([neg_df, neu_df])

# Shuffle the combined DataFrame
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean the text data
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'https?://\S+', '', text)  # Remove the hyper link
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = text.lower()  # Convert to lower case
    return text

df_sampled['text'] = df_sampled['text'].apply(clean_text)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(df_sampled['text'].values)
X = tokenizer.texts_to_sequences(df_sampled['text'].values)
X = pad_sequences(X, maxlen=50)

# One-hot encode the target
Y = pd.get_dummies(df_sampled['target']).values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))  # 3 units for 3 classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with time measurement
batch_size = 64
epochs = 5

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)

# Save the entire model
model.save('model.h5')

# Evaluate the model
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("Test score:", score)
print("Test accuracy:", acc)

# Load the model back
model = tf.keras.models.load_model('model.h5')
