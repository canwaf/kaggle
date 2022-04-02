# %%
# %%
import tensorflow as tf
import pandas as pd
import numpy as np

# %%
# Data cleaning
def extract_data(data: pd.DataFrame) -> pd.DataFrame:
    # Name cleanup to get features of Title, Forename and Lastname
    data[["LastNm", "Name"]] = data["Name"].str.split(",", expand=True)
    data[["Title", "ForeNm"]] = data["Name"].str.split(
        ". ", n=1, expand=True, regex=False
    )
    data = data.drop(["Name"], axis=1)
    data["Sex"] = data["Sex"].astype("category").cat.codes

    obj_col = data.select_dtypes(["object"])
    data[obj_col.columns] = obj_col.apply(lambda x: x.str.strip())

    data["Rooms"] = (
        data["Cabin"]
        .fillna("Steerage")
        .str.split(" ", expand=False)
        .map(lambda x: 0 if x == ["Steerage"] else len(x))
    )

    data["Floor"] = data["Cabin"].fillna("Steerage").str[:1]

    data = data.drop(columns=["Ticket", "ForeNm", "LastNm", "PassengerId", "Cabin"])

    for c in data.columns:
      data[c] = data[c].fillna(data[c].mode()[0])

    return data

# %%
df = extract_data(pd.read_csv("train.csv"))
df

# %%
# Preprocessing 
categorical_features_names = ["Embarked", "Title", "Floor", "Pclass"]
numeric_features_names = ["SibSp", "Age", "Parch", "Fare", "Rooms"]
binary_features_names = ["Sex"]

def create_preprocessor(categorical_features_names: list, numeric_features_names: list, binary_features_names: list):


    inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif (name in categorical_features_names or
                name in binary_features_names):
            dtype = tf.int64
        elif (name in numeric_features_names):
            dtype = tf.float32
        else:
            continue

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)
    inputs

    for name in binary_features_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = tf.cast(inp, tf.float32)
        preprocessed.append(float_value)

inputs = create_preprocessor(categorical_features_names=categorical_features_names, numeric_features_names=numeric_features_names, binary_features_names=binary_features_names)
# %%
survived = df.pop('Survived').to_numpy(dtype=np.int64)

# %%
preprocessed = []

for name in binary_features_names:
  inp = inputs[name]
  inp = inp[:, tf.newaxis]
  float_value = tf.cast(inp, tf.float32)
  preprocessed.append(float_value)

preprocessed

# %% [markdown]
# # Numeric Features

# %%
def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

# %%
numeric_features = df[numeric_features_names]
numeric_features.head()

# %%
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

# %%
numeric_inputs = {}
for name in numeric_features:
  numeric_inputs[name]=inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

preprocessed

# %% [markdown]
# # Categorical features

# %%
for name in categorical_features_names:
  vocab = sorted(set(df[name].dropna().values))
  print(f'name: {name}')
  print(f'vocab: {vocab}\n')

  if type(vocab[0]) is str:
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
  else:
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

  x = inputs[name][:, tf.newaxis]
  x = lookup(x)
  preprocessed.append(x)

# %%
preprocessed

# %% [markdown]
# 

# %%
preprocesssed_result = tf.concat(preprocessed, axis=-1)
preprocesssed_result

# %%
preprocessor = tf.keras.Model(inputs, preprocesssed_result)

# %%
tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True)

# %%
preprocessor(dict(df.iloc[:1]))

# %%
body = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)


# %%
inputs

# %%
x = preprocessor(inputs)
x

# %%
result = body(x)
result

# %%
model = tf.keras.Model(inputs, result)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

# %%
model.summary()

# %%
EPOCHS = 100
BATCH_SIZE = 20

history = model.fit(dict(df), survived, epochs=EPOCHS, batch_size=BATCH_SIZE)

# %%
out_df = extract_data(pd.read_csv("test.csv"))
# %%

model.fit()