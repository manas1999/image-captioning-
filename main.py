{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.utils import shuffle\n",
    "import re\n",
    "import numpy as np\n",
    "import os\n",
    "import time\n",
    "import json\n",
    "from glob import glob\n",
    "from PIL import Image\n",
    "import pickle\n",
    "from tqdm import tqdm\n",
    "from nltk.translate.bleu_score import sentence_bleu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "annotation_file = './annotations/captions_train2014.json'\n",
    "PATH = './train2014/'\n",
    "with open(annotation_file, 'r') as f:\n",
    "    annotations = json.load(f)\n",
    "\n",
    "all_captions = []\n",
    "all_img_name_vector = []\n",
    "\n",
    "for annot in annotations['annotations']:\n",
    "    caption = '<start> ' + annot['caption'] + ' <end>'\n",
    "    image_id = annot['image_id']\n",
    "    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)\n",
    "    if os.path.exists(PATH + 'COCO_train2014_' + '%012d.npy' % (image_id)):\n",
    "        all_img_name_vector.append(full_coco_image_path)\n",
    "        all_captions.append(caption)\n",
    "\n",
    "train_captions, img_name_vector = shuffle(all_captions,\n",
    "                                          all_img_name_vector,\n",
    "                                          random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "81219.0\n"
     ]
    }
   ],
   "source": [
    "print(len(img_name_vector)/5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<start> A vase is holding the 1st place winning flowers. <end>\n",
      "./train2014/COCO_train2014_000000073196.jpg\n"
     ]
    }
   ],
   "source": [
    "print(train_captions[1])\n",
    "print(img_name_vector[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calc_max_length(tensor):\n",
    "    return max(len(t) for t in tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=\"<unk>\",filters='!\"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')\n",
    "tokenizer.fit_on_texts(train_captions)\n",
    "train_seqs = tokenizer.texts_to_sequences(train_captions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer.word_index['<pad>'] = 0\n",
    "tokenizer.index_word[0] = '<pad>'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_seqs = tokenizer.texts_to_sequences(train_captions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "52\n"
     ]
    }
   ],
   "source": [
    "max_length = calc_max_length(train_seqs)\n",
    "print(max_length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_name_train, img_name_test, cap_train, cap_test = train_test_split(img_name_vector,\n",
    "                                                                    cap_vector,\n",
    "                                                                    test_size=0.1,\n",
    "                                                                    random_state=0)\n",
    "\n",
    "img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_train,\n",
    "                                                                    cap_train,\n",
    "                                                                    test_size=0.11111,\n",
    "                                                                    random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(406095, 406095, 324875, 324875, 40610, 40610, 40610, 40610)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(img_name_vector),len(cap_vector),len(img_name_train), len(cap_train), len(img_name_val), len(cap_val),len(img_name_test), len(cap_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 64\n",
    "BUFFER_SIZE = 1000\n",
    "embedding_dim = 256\n",
    "units = 512\n",
    "vocab_size = len(tokenizer.word_index) + 1\n",
    "steps_per_epoch = len(img_name_train) // BATCH_SIZE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "23700"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(tokenizer.word_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "def map_func(img_name, cap):\n",
    "    img_tensor = np.load(img_name.decode('utf-8')[:-4]+'.npy')\n",
    "    np.float32(img_tensor)\n",
    "    img_tensor = img_tensor.reshape((img_tensor.shape[0], 7*7*256))\n",
    "    new_matrix =  np.zeros((39,7*7*256), dtype=\"float32\")\n",
    "    new_matrix[:img_tensor.shape[0],:] = img_tensor\n",
    "    return new_matrix, cap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before flag parsing goes to stderr.\n",
      "W0609 20:12:30.362071 140612036593472 deprecation.py:323] From /home/billzhang/anaconda3/envs/test2/lib/python3.7/site-packages/tensorflow/python/ops/script_ops.py:476: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "tf.py_func is deprecated in TF V2. Instead, there are two\n",
      "    options available in V2.\n",
      "    - tf.py_function takes a python function which manipulates tf eager\n",
      "    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to\n",
      "    an ndarray (just call tensor.numpy()) but having access to eager tensors\n",
      "    means `tf.py_function`s can use accelerators such as GPUs as well as\n",
      "    being differentiable using a gradient tape.\n",
      "    - tf.numpy_function maintains the semantics of the deprecated tf.py_func\n",
      "    (it is not differentiable, and manipulates numpy arrays). It drops the\n",
      "    stateful argument making all functions stateful.\n",
      "    \n"
     ]
    }
   ],
   "source": [
    "dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))\n",
    "\n",
    "# using map to load the numpy files in parallel\n",
    "dataset = dataset.map(lambda item1, item2: tf.numpy_function(\n",
    "          map_func, [item1, item2], [tf.float32, tf.int32]),\n",
    "          num_parallel_calls=tf.data.experimental.AUTOTUNE)\n",
    "\n",
    "# shuffling and batching\n",
    "dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)\n",
    "dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(tf.keras.Model):\n",
    "  def __init__(self, embedding_dim, enc_units, batch_sz):\n",
    "    super(Encoder, self).__init__()\n",
    "    self.batch_sz = batch_sz\n",
    "    self.enc_units = enc_units\n",
    "    self.fc = tf.keras.layers.Dense(embedding_dim)\n",
    "    self.gru = tf.keras.layers.GRU(self.enc_units,\n",
    "                                   return_sequences=True,\n",
    "                                   return_state=True,\n",
    "                                   recurrent_initializer='glorot_uniform')\n",
    "\n",
    "  def call(self, x, hidden):\n",
    "    x = self.fc(x)\n",
    "    x = tf.nn.relu(x)\n",
    "    output, state = self.gru(x, initial_state = hidden)\n",
    "    return output, state\n",
    "\n",
    "  def initialize_hidden_state(self):\n",
    "    return tf.zeros((self.batch_sz, self.enc_units))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(TensorShape([64, 39, 12544]), TensorShape([64, 52]))"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "example_input_batch, example_target_batch = next(iter(dataset))\n",
    "example_input_batch.shape, example_target_batch.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = Encoder(embedding_dim, units, BATCH_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Encoder output shape: (batch size, sequence length, units) (64, 39, 512)\n",
      "Encoder Hidden state shape: (batch size, units) (64, 512)\n"
     ]
    }
   ],
   "source": [
    "sample_hidden = encoder.initialize_hidden_state()\n",
    "sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)\n",
    "print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))\n",
    "print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "class BahdanauAttention(tf.keras.Model):\n",
    "  def __init__(self, units):\n",
    "    super(BahdanauAttention, self).__init__()\n",
    "    self.W1 = tf.keras.layers.Dense(units)\n",
    "    self.W2 = tf.keras.layers.Dense(units)\n",
    "    self.V = tf.keras.layers.Dense(1)\n",
    "\n",
    "  def call(self, query, values):\n",
    "    hidden_with_time_axis = tf.expand_dims(query, 1)\n",
    "    score = self.V(tf.nn.tanh(\n",
    "        self.W1(values) + self.W2(hidden_with_time_axis)))\n",
    "    attention_weights = tf.nn.softmax(score, axis=1)\n",
    "    context_vector = attention_weights * values\n",
    "    context_vector = tf.reduce_sum(context_vector, axis=1)\n",
    "\n",
    "    return context_vector, attention_weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Attention result shape: (batch size, units) (64, 512)\n",
      "Attention weights shape: (batch_size, sequence_length, 1) (64, 39, 1)\n"
     ]
    }
   ],
   "source": [
    "attention_layer = BahdanauAttention(10)\n",
    "attention_result, attention_weights = attention_layer(sample_hidden, sample_output)\n",
    "\n",
    "print(\"Attention result shape: (batch size, units) {}\".format(attention_result.shape))\n",
    "print(\"Attention weights shape: (batch_size, sequence_length, 1) {}\".format(attention_weights.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RNN_Decoder(tf.keras.Model):\n",
    "    def __init__(self, embedding_dim, units, vocab_size):\n",
    "        super(RNN_Decoder, self).__init__()\n",
    "        self.units = units\n",
    "\n",
    "        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)\n",
    "        self.gru = tf.keras.layers.GRU(self.units,\n",
    "                                       return_sequences=True,\n",
    "                                       return_state=True,\n",
    "                                       recurrent_initializer='glorot_uniform')\n",
    "        self.fc1 = tf.keras.layers.Dense(self.units)\n",
    "        self.fc2 = tf.keras.layers.Dense(vocab_size)\n",
    "\n",
    "        self.attention = BahdanauAttention(self.units)\n",
    "        self.softmax =tf.keras.layers.Softmax()\n",
    "    \n",
    "    def call(self, x, hidden, enc_output):\n",
    "        context_vector, attention_weights = self.attention(hidden, enc_output)\n",
    "        x = self.embedding(x)\n",
    "        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)\n",
    "        output, state = self.gru(x)\n",
    "        output = self.fc1(output)\n",
    "        output = tf.reshape(output, (-1, output.shape[2]))\n",
    "        x = self.fc2(output)\n",
    "        sm = self.softmax(x)\n",
    "        return sm, x, state, attention_weights\n",
    "\n",
    "    def reset_state(self, batch_size):\n",
    "        return tf.zeros((batch_size, self.units))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TensorShape([64, 512])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "decoder = RNN_Decoder(embedding_dim, units, vocab_size)\n",
    "hidden = decoder.reset_state(batch_size=64)\n",
    "hidden.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "_, sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),\n",
    "                                      sample_hidden, sample_output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decoder output shape: (batch_size, vocab size) (64, 23701)\n"
     ]
    }
   ],
   "source": [
    "print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = tf.keras.optimizers.Adam(lr=0.001)\n",
    "loss_object = tf.keras.losses.SparseCategoricalCrossentropy(\n",
    "    from_logits=True, reduction='none')\n",
    "\n",
    "def loss_function(real, pred):\n",
    "  mask = tf.math.logical_not(tf.math.equal(real, 0))\n",
    "  loss_ = loss_object(real, pred)\n",
    "\n",
    "  mask = tf.cast(mask, dtype=loss_.dtype)\n",
    "  loss_ *= mask\n",
    "\n",
    "  return tf.reduce_mean(loss_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Adam',\n",
       " 'learning_rate': 0.001,\n",
       " 'decay': 0.0,\n",
       " 'beta_1': 0.9,\n",
       " 'beta_2': 0.999,\n",
       " 'epsilon': 1e-07,\n",
       " 'amsgrad': False}"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "optimizer.get_config()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoint_dir = './training_checkpoints'\n",
    "checkpoint_prefix = os.path.join(checkpoint_dir, \"ckpt\")\n",
    "checkpoint = tf.train.Checkpoint(optimizer=optimizer,\n",
    "                                 encoder=encoder,\n",
    "                                 decoder=decoder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def train_step(inp, targ, enc_hidden):\n",
    "  loss = 0\n",
    "\n",
    "  with tf.GradientTape() as tape:\n",
    "    enc_output, enc_hidden = encoder(inp, enc_hidden)\n",
    "\n",
    "    dec_hidden = enc_hidden\n",
    "\n",
    "    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)\n",
    "\n",
    "    for t in range(1, targ.shape[1]):\n",
    "      _, predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)\n",
    "      lossa = loss_function(targ[:, t], predictions)\n",
    "      train_loss(lossa)\n",
    "      train_accuracy(targ[:, t], predictions)\n",
    "      loss += lossa\n",
    "\n",
    "      dec_input = tf.expand_dims(targ[:, t], 1)\n",
    "    \n",
    "\n",
    "  batch_loss = (loss / int(targ.shape[1]))\n",
    "  \n",
    "\n",
    "  variables = encoder.trainable_variables + decoder.trainable_variables\n",
    "\n",
    "  gradients = tape.gradient(loss, variables)\n",
    "\n",
    "  optimizer.apply_gradients(zip(gradients, variables))\n",
    "\n",
    "  return batch_loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime\n",
    "current_time = datetime.datetime.now().strftime(\"%Y%m%d-%H%M%S\")\n",
    "train_log_dir = 'logs6/gradient_tape/' + current_time + '/train'\n",
    "train_summary_writer = tf.summary.create_file_writer(train_log_dir)\n",
    "test_log_dir = 'logs6/gradient_tape/' + current_time + '/test'\n",
    "test_summary_writer = tf.summary.create_file_writer(test_log_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)\n",
    "train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')\n",
    "test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)\n",
    "test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f3099d48c50>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "encoder.load_weights('./weights/encoder_checkpoint6')\n",
    "decoder.load_weights('./weights/decoder_checkpoint6')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1 Batch 0 Loss 0.5331\n",
      "Epoch 1 Batch 100 Loss 0.4799\n",
      "Epoch 1 Batch 200 Loss 0.5259\n",
      "Epoch 1 Batch 300 Loss 0.5364\n",
      "Epoch 1 Batch 400 Loss 0.5091\n",
      "Epoch 1 Batch 500 Loss 0.5563\n",
      "Epoch 1 Batch 600 Loss 0.5624\n",
      "Epoch 1 Batch 700 Loss 0.5454\n",
      "Epoch 1 Batch 800 Loss 0.5551\n",
      "Epoch 1 Batch 900 Loss 0.5490\n",
      "Epoch 1 Batch 1000 Loss 0.5017\n",
      "Epoch 1 Batch 1100 Loss 0.5782\n",
      "Epoch 1 Batch 1200 Loss 0.5081\n",
      "Epoch 1 Batch 1300 Loss 0.5164\n",
      "Epoch 1 Batch 1400 Loss 0.5552\n",
      "Epoch 1 Batch 1500 Loss 0.5072\n",
      "Epoch 1 Batch 1600 Loss 0.5885\n",
      "Epoch 1 Batch 1700 Loss 0.4814\n",
      "Epoch 1 Batch 1800 Loss 0.5724\n",
      "Epoch 1 Batch 1900 Loss 0.5149\n",
      "Epoch 1 Batch 2000 Loss 0.5874\n",
      "Epoch 1 Batch 2100 Loss 0.5635\n",
      "Epoch 1 Batch 2200 Loss 0.5380\n",
      "Epoch 1 Batch 2300 Loss 0.5278\n",
      "Epoch 1 Batch 2400 Loss 0.5319\n",
      "Epoch 1 Batch 2500 Loss 0.5214\n",
      "Epoch 1 Batch 2600 Loss 0.5399\n",
      "Epoch 1 Batch 2700 Loss 0.5809\n",
      "Epoch 1 Batch 2800 Loss 0.5385\n",
      "Epoch 1 Batch 2900 Loss 0.4966\n",
      "Epoch 1 Batch 3000 Loss 0.5244\n",
      "Epoch 1 Batch 3100 Loss 0.5355\n",
      "Epoch 1 Batch 3200 Loss 0.4967\n",
      "Epoch 1 Batch 3300 Loss 0.4974\n",
      "Epoch 1 Batch 3400 Loss 0.5186\n",
      "Epoch 1 Batch 3500 Loss 0.5492\n",
      "Epoch 1 Batch 3600 Loss 0.4916\n",
      "Epoch 1 Batch 3700 Loss 0.5092\n",
      "Epoch 1 Batch 3800 Loss 0.5272\n",
      "Epoch 1 Batch 3900 Loss 0.5520\n",
      "Epoch 1 Batch 4000 Loss 0.5882\n",
      "Epoch 1 Batch 4100 Loss 0.5210\n",
      "Epoch 1 Batch 4200 Loss 0.5438\n",
      "Epoch 1 Batch 4300 Loss 0.5428\n",
      "Epoch 1 Batch 4400 Loss 0.5266\n",
      "Epoch 1 Batch 4500 Loss 0.5013\n",
      "Epoch 1 Batch 4600 Loss 0.4963\n",
      "Epoch 1 Batch 4700 Loss 0.5026\n",
      "Epoch 1 Batch 4800 Loss 0.5600\n",
      "Epoch 1 Batch 4900 Loss 0.5122\n",
      "Epoch 1 Batch 5000 Loss 0.5170\n",
      "Epoch 1 Loss 0.5310\n",
      "Time taken for 1 epoch 1715.255756855011 sec\n",
      "\n"
     ]
    }
   ],
   "source": [
    "EPOCHS = 1\n",
    "for epoch in range(EPOCHS):\n",
    "  start = time.time()\n",
    "\n",
    "  enc_hidden = encoder.initialize_hidden_state()\n",
    "  total_loss = 0\n",
    "\n",
    "  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):\n",
    "    batch_loss = train_step(inp, targ, enc_hidden)\n",
    "    total_loss += batch_loss\n",
    "    with train_summary_writer.as_default():\n",
    "        tf.summary.scalar('loss', train_loss.result(), step=epoch)\n",
    "\n",
    "    if batch % 100 == 0:\n",
    "        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,\n",
    "                                                     batch,\n",
    "                                                     batch_loss.numpy()))\n",
    "  if (epoch + 1) % 2 == 0:\n",
    "    checkpoint.save(file_prefix = checkpoint_prefix)\n",
    "    encoder.save_weights('./weights/encoder_checkpoint6')\n",
    "    decoder.save_weights('./weights/decoder_checkpoint6')\n",
    "\n",
    "  print('Epoch {} Loss {:.4f}'.format(epoch + 1,\n",
    "                                      total_loss / steps_per_epoch))\n",
    "  print('Time taken for 1 epoch {} sec\\n'.format(time.time() - start))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder.save_weights('./weights/encoder_checkpoint6')\n",
    "decoder.save_weights('./weights/decoder_checkpoint6')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "def greedy_search(image_name):\n",
    "    img_tensor = np.load(image_name[:-4]+'.npy')\n",
    "    np.float32(img_tensor)\n",
    "    img_tensor = img_tensor.reshape((img_tensor.shape[0], 7*7*256))\n",
    "    new_matrix =  np.zeros((39,7*7*256), dtype=\"float32\")\n",
    "    new_matrix[:img_tensor.shape[0],:] = img_tensor\n",
    "\n",
    "\n",
    "    inputs = tf.convert_to_tensor(new_matrix)\n",
    "    result = ''\n",
    "    inputs = tf.expand_dims(inputs, 0)\n",
    "    inputs.shape\n",
    "\n",
    "\n",
    "    hidden = tf.zeros((1, units))\n",
    "    hidden.shape\n",
    "    enc_out, enc_hidden = encoder(inputs, hidden)\n",
    "    dec_hidden = enc_hidden\n",
    "    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)\n",
    "\n",
    "    for t in range(max_length):\n",
    "            _, predictions, dec_hidden, attention_weights = decoder(dec_input,\n",
    "                                                                 dec_hidden,\n",
    "                                                                 enc_out)\n",
    "\n",
    "\n",
    "            predicted_id = tf.argmax(predictions[0]).numpy()\n",
    "            if tokenizer.index_word[predicted_id] == '<end>':\n",
    "                break\n",
    "\n",
    "            result += tokenizer.index_word[predicted_id] + ' '\n",
    "\n",
    "            \n",
    "\n",
    "            dec_input = tf.expand_dims([predicted_id], 0)\n",
    "    return result[:-1]\n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_ref = {}\n",
    "for annot in annotations['annotations']:\n",
    "    caption = '<start> ' + annot['caption'] + ' <end>'\n",
    "    image_id = annot['image_id']\n",
    "    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)\n",
    "    if os.path.exists(PATH + 'COCO_train2014_' + '%012d.npy' % (image_id)):\n",
    "        if full_coco_image_path not in all_ref:\n",
    "            all_ref[full_coco_image_path] = [caption.split()]\n",
    "        else:\n",
    "            all_ref[full_coco_image_path].append(caption.split())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calc_length(list):\n",
    "    length = 0\n",
    "    for l in list:\n",
    "        if l != 4:\n",
    "            length += 1\n",
    "        elif l == 4:\n",
    "            break\n",
    "    return length\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "def beam_search(image_name, beam_index,beam_alpha=0.7):\n",
    "    debug = False\n",
    "    #image_name = img_name_train[rid]\n",
    "    img_tensor = np.load(image_name[:-4]+'.npy')\n",
    "    np.float32(img_tensor)\n",
    "    img_tensor = img_tensor.reshape((img_tensor.shape[0], 7*7*256))\n",
    "    new_matrix =  np.zeros((39,7*7*256), dtype=\"float32\")\n",
    "    new_matrix[:img_tensor.shape[0],:] = img_tensor\n",
    "\n",
    "    inputs = tf.convert_to_tensor(new_matrix)\n",
    "    inputs = tf.expand_dims(inputs, 0)\n",
    "    hidden = tf.zeros((1, units))\n",
    "\n",
    "    enc_out, enc_hidden = encoder(inputs, hidden)\n",
    "    dec_hidden = enc_hidden\n",
    "    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)\n",
    "\n",
    "    start = [tokenizer.word_index['<start>']]\n",
    "\n",
    "    start_word = [[start, 0.0, dec_hidden]]\n",
    "    total = []\n",
    "    \n",
    "    \n",
    "    for itter in range(max_length):\n",
    "    #while len(estart_word[0]) < max_length:\n",
    "        temp = []\n",
    "        for s in start_word:\n",
    "            if debug:\n",
    "                print(\"Debug:\",len(s[0]),\"Word\",s[0],\"Prob\",s[1])\n",
    "            \n",
    "            #for s in start_word:\n",
    "            if (s[0][-1] != tokenizer.word_index['<end>']):\n",
    "                dec_input = tf.expand_dims([s[0][-1]], 0)\n",
    "                softmax, predictions, dec_hidden, attention_weights = decoder(dec_input,s[2],enc_out)\n",
    "                word_preds = tf.argsort(softmax[0],direction = 'DESCENDING')[:beam_index].numpy()\n",
    "                for w in word_preds:\n",
    "                    next_cap, prob = s[0][:], s[1]\n",
    "                    next_cap.append(w)\n",
    "                    prob += np.log(softmax[0][w].numpy())\n",
    "                    #prob *= softmax[0][w].numpy()\n",
    "                    temp.append([next_cap, prob,dec_hidden])\n",
    "            else:\n",
    "                temp.append(s)\n",
    "                \n",
    "            \n",
    "        start_word = temp\n",
    "        \n",
    "        for ind in start_word:\n",
    "            if debug:\n",
    "                print(\"*Debug:\",len(ind[0]),\"Word\",ind[0],\"Prob\",ind[1])\n",
    "        \n",
    "        # Sorting according to the probabilities\n",
    "        #start_word = sorted(start_word, reverse=False, key=lambda l: l[1]*(1/(len(l[0])**0.7)))\n",
    "        #for s in start_word:\n",
    "        #    s[1] =  s[1]*(1/(calc_length(s[0])**beam_alpha))\n",
    "        \n",
    "        start_word = sorted(start_word, reverse=True, key=lambda l: l[1]*(1/(len(l[0])**beam_alpha)))\n",
    "        #total = total + start_word\n",
    "        start_word = start_word[:beam_index]\n",
    "        \n",
    "        for ind in start_word:\n",
    "            if debug:\n",
    "                print(\"**Debug:\",len(ind[0]),\"Word\",ind[0],\"Prob\",ind[1])                    \n",
    "                            \n",
    "                            \n",
    "        #print(start_word)\n",
    "    total = start_word[0][0]\n",
    "    #total = sorted(start_word, reverse=True, key=lambda l: l[1]*(1/(calc_length(l[0])**beam_alpha)))[0][0]\n",
    "    #print (total)\n",
    "    #total = total[-2][0]\n",
    "    intermediate_caption = [tokenizer.index_word[i] for i in total]\n",
    "\n",
    "    final_caption = []\n",
    "\n",
    "    for i in intermediate_caption:\n",
    "        if i != '<end>':\n",
    "            final_caption.append(i)\n",
    "        else:\n",
    "            break\n",
    "    \n",
    "    final_caption = ' '.join(final_caption[1:])\n",
    "    print (\"beam search: k = {}\".format(beam_index))\n",
    "    print (final_caption)\n",
    "    return final_caption"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_real_caption(image_name):\n",
    "    print(\"real_captions:\")\n",
    "    refs = all_ref[image_name]\n",
    "    for ref in refs:\n",
    "        print (' '.join(ref))\n",
    "    return refs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Bleu Score\n",
    "def calcBleuScore(image_name, sentence):\n",
    "    debug = False\n",
    "    captions = all_ref[image_name]\n",
    "    reference = []\n",
    "    for c in captions:\n",
    "        c = \" \".join(c)\n",
    "        inp_list = re.sub(\"[^<*\\w>*]\", \" \", c).lower().split()\n",
    "        reference.append(inp_list[1:-1])\n",
    "    candidate = re.sub(\"[^\\w]\", \" \", sentence).lower().split()\n",
    "    #test cases\n",
    "    #reference = [['this', 'is', 'a', 'test']]\n",
    "    #reference = [['this', 'is','test']]\n",
    "    #reference = [['this', 'is', 'test'],['this', 'is','a','test']]\n",
    "    #candidate = ['this', 'is', 'a', 'test']\n",
    "\n",
    "    bleu_score = np.zeros([4,1])    \n",
    "\n",
    "    bleu_score[0] = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))\n",
    "    bleu_score[1] = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))\n",
    "    bleu_score[2] = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))\n",
    "    bleu_score[3] = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))\n",
    "    \n",
    "#     if (debug):\n",
    "#         print('Reference:\\n', reference)\n",
    "#         print('Candidate:\\n', candidate)\n",
    "#         print('*********************************************************************************')\n",
    "    print('Individual 1-gram: %.3f' % bleu_score[0])\n",
    "    print('Individual 2-gram: %.3f' % bleu_score[1])\n",
    "    print('Individual 3-gram: %.3f' % bleu_score[2])\n",
    "    print('Individual 4-gram: %.3f' % bleu_score[3])\n",
    "        \n",
    "    return bleu_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "./train2014/COCO_train2014_000000169174.jpg\n",
      "(3, 7, 7, 256)\n",
      "real_captions:\n",
      "<start> a black and white photo of a parked motorcycle <end>\n",
      "<start> A motorcycle that is parked and standing on a street. <end>\n",
      "<start> A black and white motorcycle parked on concrete. <end>\n",
      "<start> A white and black motorcycle parked in the sun. <end>\n",
      "<start> A motorcycle that is parked on some pavement. <end>\n",
      "beam search: k = 2\n",
      "a motorcycle is parked on the side of the road\n",
      "beam search: k = 3\n",
      "a motorcycle parked on the side of a road\n",
      "beam search: k = 5\n",
      "a motorcycle parked on the side of a road\n",
      "beam search: k = 8\n",
      "a motorcycle parked on the side of a road\n",
      "beam search: k = 10\n",
      "a motorcycle parked on the side of a road\n",
      "Individual 1-gram: 0.778\n",
      "Individual 2-gram: 0.500\n",
      "Individual 3-gram: 0.143\n",
      "Individual 4-gram: 0.000\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/billzhang/anaconda3/envs/test2/lib/python3.7/site-packages/nltk/translate/bleu_score.py:523: UserWarning: \n",
      "The hypothesis contains 0 counts of 4-gram overlaps.\n",
      "Therefore the BLEU score evaluates to 0, independently of\n",
      "how many N-gram overlaps of lower order it contains.\n",
      "Consider using lower n-gram order or use SmoothingFunction()\n",
      "  warnings.warn(_msg)\n"
     ]
    },
    {
     "data": {
      "image/png": "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
