

################################################################################
#                        MUSIC GENERATION FULL CODE                            #
################################################################################

try:
  import fluidsynth
except ImportError:
  !pip install fluidsynth
  import fluidsynth

try:
  import pretty_midi
except ImportError:
  !pip install pretty_midi
  import pretty_midi

import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
import keras


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
seq_length = 25
vocab_size = 128
key_order = ['pitch', 'step', 'duration']


# convert the data in midi format to notes to feed them as input
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  # convert midi to nodes
  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

# create the notes from a randomly choosen music in midi files so that each time a new note sequence will be given as input and new music will be generated
def create_raw_notes(emotion):

  # take a music randomly from this location
  data_dir = pathlib.Path("/content/drive/MyDrive/music_dataset_2/midi_files/"+emotion+"/")
  filenames = glob.glob(str(data_dir/'**/*.mid*'))
  print('Number of files:', len(filenames))

  sample_file = filenames[1]
  print(sample_file)

  # use pretty_midi library to handle midi files
  pm = pretty_midi.PrettyMIDI(sample_file)

  print('Number of instruments:', len(pm.instruments))
  instrument = pm.instruments[0]
  instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
  print("Instruments", pm.instruments)
  print('Instrument name:', instrument_name)


  for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name},'
          f' duration={duration:.4f}')

  # convert midi fata to notes
  raw_notes = midi_to_notes(sample_file)
  raw_notes.head()

  return raw_notes

# convert notes to midi data to again generate a music from predicted inputs
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
  ) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


# create train set based on the emotion
def create_train_set(emotion):

  # directory of midi files based on emotions, each midi files are located at specific folder based on emotion
  data_dir = pathlib.Path("/content/drive/MyDrive/music_dataset_2/midi_files/"+emotion+"/")
  filenames = glob.glob(str(data_dir/'**/*.mid*'))
  print('Number of files:', len(filenames))

  sample_file = filenames[1]
  print(sample_file)

  pm = pretty_midi.PrettyMIDI(sample_file)

  print('Number of instruments:', len(pm.instruments))
  instrument = pm.instruments[0]
  instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
  print("Instruments", pm.instruments)
  print('Instrument name:', instrument_name)


  for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name},'
          f' duration={duration:.4f}')

  raw_notes = midi_to_notes(sample_file)
  raw_notes.head()


  get_note_names = np.vectorize(pretty_midi.note_number_to_name)
  sample_note_names = get_note_names(raw_notes['pitch'])
  sample_note_names[:10]

  # pass 50 files to create train set and append all notes to all_notes list
  num_files = 50
  all_notes = []
  for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

  all_notes = pd.concat(all_notes)

  n_notes = len(all_notes)
  print('Number of notes parsed:', n_notes)


  train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

  # convert train set to tensor structure
  notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
  notes_ds.element_spec

  # create train input and labels for the model
  seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
  seq_ds.element_spec

  for seq, target in seq_ds.take(1):
    print('sequence shape:', seq.shape)
    print('sequence elements (first 10):', seq[0: 10])
    print()
    print('target:', target)


  batch_size = 64
  buffer_size = n_notes - seq_length  # the number of items in the dataset
  train_ds = (seq_ds
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))



  train_ds.element_spec

  return train_ds,raw_notes



# convert notes to midi data to again generate a music from predicted inputs
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
  ) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm




# construct the train set by using window method
def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size = 128,) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)
  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

# Normalize note pitch
def scale_pitch(x):
  x = x/[vocab_size,1.0,1.0]
  return x

# Split the labels
def split_labels(sequences):
  inputs = sequences[:-1]
  labels_dense = sequences[-1]
  labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

  return scale_pitch(inputs), labels


# create the RNN model based on the train set which was derived from spesific emotion
def model_RNN(train_ds):
  input_shape = (seq_length, 3)
  learning_rate = 0.005

  inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.LSTM(128)(inputs) # add RNN layer to NN

  # output is pitch, step and duration for each note
  outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
  }


  model = tf.keras.Model(inputs, outputs)

  loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
  }

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss=loss, optimizer=optimizer)

  model.summary()


  losses = model.evaluate(train_ds, return_dict=True)
  #losses


  model.compile(
      loss=loss,
      loss_weights={
          'pitch': 0.05,
          'step': 1.0,
          'duration':1.0,
      },
      optimizer=optimizer,
  )


  model.evaluate(train_ds, return_dict=True)

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath='./training_checkpoints/ckpt_{epoch}',
          save_weights_only=True),
      tf.keras.callbacks.EarlyStopping(
          monitor='loss',
          patience=5,
          verbose=1,
          restore_best_weights=True),
  ]


  epochs = 5

  history = model.fit(
      train_ds,
      epochs=epochs,
      callbacks=callbacks,
  )

  #plt.plot(history.epoch, history.history['loss'], label='total loss')
  #plt.show()

  return model

@keras.saving.register_keras_serializable()
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)


# predict the next note given a sequence of notes
def predict_next_note(
    notes: np.ndarray,
    model: tf.keras.Model,
    temperature: float = 1.0) -> tuple[int, float, float]:
  """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)


# generate the music from the saved model, the notes of randomly choosen music will be given as input to this function
def generate_music(raw_notes, model):

  temperature = 2
  num_predictions = 120

  sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

  # The initial sequence of notes; pitch is normalized similar to training
  # sequences
  input_notes = (
      sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

  generated_notes = []
  prev_start = 0
  for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    ################# THIS IS A CRUICAL PART SINCE I EXPAND THE DURATION AND STEP TO GENERATE A MORE COMPREHENSIBLE MUSIC, OTHERWISE IT SOMETIMES GENERATES MUSICS AS 1 SECOND LONG
    step, duration = step*10, duration*10
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

  generated_notes = pd.DataFrame(
      generated_notes, columns=(*key_order, 'start', 'end'))



  generated_notes.head(10)

  out_file = 'output.mid'
  out_pm = notes_to_midi(
      generated_notes, out_file=out_file, instrument_name="Acoustic Grand Piano")#instrument_name=instrument_name
  #display_audio(out_pm)

  from google.colab import files
  files.download(out_file)  # download the midi file to computer


#CREATE TRAIN SET AND RNN MODEL FOR EACH EMOTION
train_set_happy, raw_notes_happy = create_train_set("happy")
model_happy = model_RNN(train_set_happy)
train_set_sad, raw_notes_sad  = create_train_set("sad")
model_sad = model_RNN(train_set_sad)
train_set_angry, raw_notes_angry  = create_train_set("angry")
model_angry = model_RNN(train_set_angry)
train_set_relax, raw_notes_relax  = create_train_set("relax")
model_relax = model_RNN(train_set_relax)


############## YOU CAN DELETE THE COMMENT SIGNS IN BELOW LINES TO SAVE THE NEWLY TRAINED MODEL FOR LATER USE

# import numpy as np
# import keras
#save the model for later use
# saved_model_path_happy  = '/content/drive/MyDrive/keras_models/music_generator_happy.keras'
# saved_model_path_sad  = '/content/drive/MyDrive/keras_models/music_generator_sad.keras'
# saved_model_path_angry  = '/content/drive/MyDrive/keras_models/music_generator_angry.keras'
# saved_model_path_relax  = '/content/drive/MyDrive/keras_models/music_generator_relax.keras'
# model_happy.save(saved_model_path_happy)
# model_sad.save(saved_model_path_sad)
# model_angry.save(saved_model_path_angry)
# model_relax.save(saved_model_path_relax)