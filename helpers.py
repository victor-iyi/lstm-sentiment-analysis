import os
import sys
import re

import numpy as np


GLOVE_WORDS_FILE = 'saved/wordsList.npy'
GLOVE_VECTOR_FILE = 'saved/wordVectors.npy'

# GloVe words
GLOVE_WORDS = np.load(GLOVE_WORDS_FILE)
GLOVE_WORDS = [word.decode('UTF-8') for word in GLOVE_WORDS]
# GloVe vectors
GLOVE_VECTORS = np.load(GLOVE_VECTOR_FILE)



# Create data (Features & Labels)
class Word2ID(object):
    """
    self.__init__(self, data_dir, logging=True)
    :params data_dir: str
            Top level directory containing dataset
    :params logging: bool, default True
            Give feedback of what's happening in the background.

    -------Methods-------
    train_test_split(self, X, y, **kwargs):
        `X` features
        `y` labels
        **kwargs Keyword arguments
    get_embedding_ids(self, file_or_text):
        `file_or_text` File path or text corpus to be converted with GloVe embeddings
    next_batch(self, batch_size, shuffle=True):
        `batch_size` How big is the next batch to be generated.
        `shuffle` Randomly shuffle the next batch?
    
    -------Atrributes--------
    features
    labels
    num_examples
    label_names
    labels_one_hot
    """
    
    def __init__(self, data_dir, logging=True):
        self._data_dir = data_dir
        self.logging = logging
        
        print('Automatically determining `max_seq_len`')
        word_count = self._get_num_words()
        self.max_seq_len = int(sum(word_count) / len(word_count))
        self._num_files = len(word_count)
        print('Total word count = {:,}\nEstimated `max_seq_len` = {:,}\nnum_files = {:,}\n'.format(
                sum(word_count), self.max_seq_len, self._num_files))
        del word_count  # Free up memory
        self.__special_chars = re.compile('[^A-Za-z0-9 ]+')
        # !- label names & label one-hot
        self._one_hot_labels()
        self._features = np.ndarray(shape=[self._num_files, self.max_seq_len], dtype=int)
        self._labels = np.ndarray(shape=[self._num_files, len(self._data_folders)], dtype=int)
        counter = 0
        for i, d in enumerate(self._data_folders):
            files = [os.path.join(d, f) for f in os.listdir(d) if f[0] != '.']
            for j, f in enumerate(files):
                if self.logging:
                    sys.stdout.write('\r{:,} of {:,} labels\t{:,} of {:,} features'.format(
                                     i+1, len(self._data_folders), j+1, len(files)))
                # !- features & labels
                features = self._get_embedding_ids(f)
                labels = self._labels_one_hot[i]
                # !- Update...
                self._features[counter] = features
                self._labels[counter] = labels
                counter += 1
        # For next batch
        self._num_examples = self._features.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    # !- ============================ Class methods ============================ -! #

    def train_test_split(self, X, y, **kwargs):
        """
        Split data into training/testing and validation sets
        """
        return self._train_test_split(X, y, **kwargs)
    
    def get_embedding_ids(self, file_or_text):
        """
        Retrieve GloVe IDs for giving text or file
        """
        return self._get_embedding_ids(file_or_text)
    
    def next_batch(self, batch_size, shuffle=True):
        """
        Get the next batch for features and labels
        """
        return self._next_batch(batch_size, shuffle)
    
    # !- ============================ Properties ============================ -! #

    @property
    def features(self):
        return self._features
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def label_names(self):
        return self._label_names
    
    @property
    def labels_one_hot(self):
        return self._labels_one_hot
    
    # !- ============================ Protected ============================ -! #
    
    # !- Creates `ids` using GloVe embeddings
    def _get_embedding_ids(self, file_or_text):
        matrix_ids = np.zeros(shape=[1, self.max_seq_len])
        if os.path.isfile(file_or_text):
            words = open(file_or_text, mode='r', encoding='utf-8').read()
        else:
            words = file_or_text
        words = self._clean_words(words)
        word_split = words.split()
        # create GloVe learned embeddings
        for i, word in enumerate(word_split):
            try:
                matrix_ids[0, i] = GLOVE_WORDS.index(word)
            except:
                matrix_ids[0, i] = GLOVE_WORDS.index('unk')
            finally:
                if i >= self.max_seq_len:
                    break
        return matrix_ids
    
    # !- Create one-hot encoding
    def _one_hot(self, arr):
        # collect the lists & the uniques
        arr, unique = list(arr), list(set(arr))
        encoded = np.zeros(shape=[len(arr), len(unique)], dtype=int)
        for i, val in enumerate(arr):
            index = unique.index(val)
            encoded[i, index] = 1.
        return encoded
    
    # !- Retruns labels and labels as one-hot arrays
    def _one_hot_labels(self):
        self._label_names = [d for d in os.listdir(self._data_dir) 
                  if os.path.isdir(os.path.join(self._data_dir, d))]
        self._labels_one_hot = self._one_hot(self._label_names)
    
    # !- Generate next random batch
    def _next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            permute = np.arange(self._num_examples)
            np.random.shuffle(permute)
            self._features = self._features[permute]
            self._labels = self._labels[permute]
        # Go to next batch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_examples = self._num_examples - start
            rest_features = self._features[start:self._num_examples]
            rest_labels = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                permute = np.arange(self._num_examples)
                np.random.shuffle(permute)
                self._features = self._features[permute]
                self._labels = self._labels[permute]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_examples
            end = self._index_in_epoch
            features = np.concatenate((rest_features, self._features[start:end]))
            labels = np.concatenate((rest_labels, self._labels[start:end]))
            return features, labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]
    
    def _train_test_split(self, X, y, **kwargs):
        pass
    
    def _get_num_words(self):
        num_words = []
        self._data_folders = [os.path.join(self._data_dir, df) for df in os.listdir(self._data_dir)]
        for data_folder in self._data_folders:
            data_files = [os.path.join(data_folder, df) for df in os.listdir(data_folder) if df[0] != '.']
            for data_file in data_files:
                with open(data_file, encoding='utf-8') as f:
                    line = f.readline()
                    word_count = len(line.split())
                    num_words.append(word_count)
        return num_words
    
    # !- Filter clean words
    def _clean_words(self, string):
        string = string.lower().replace('<br />', ' ')
        return str(re.sub(self.__special_chars, '', string))

