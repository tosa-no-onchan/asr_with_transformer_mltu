'''
asr_with_transformer_mltu/tools_mltu.py
'''

import os

from glob import glob
import tensorflow as tf

import numpy as np
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from configs import ModelConfigs

"""
1. tools
"""

# https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder
def latest_checkpoint(checkpoint_dir):
    #print('checkpoint_dir:',checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        return None
    #list_of_files = glob.glob('/path/to/folder/*') # * means all if need specific format then *.csv
    list_of_files = glob(checkpoint_dir+'/*weights.h5') # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def latest_checkno(latest):
        #print('latest:',latest)
        ret = latest.split("/")[-1]
        ret = ret.split(".")[0]
        return int(ret.split("-")[-1])


"""
2. original dataset access
"""

def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    data = []
    for w in wavs:
        w=w.replace('\\', '/')
        #print("w:",w)
        id = w.split("/")[-1].split(".")[0]
        #print("id",id)
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data

"""
## Preprocess the dataset
"""
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

#def create_text_ds(data):
def create_text_ds(data,vectorizer):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds

def path_to_audio(path):
    # spectrogram using stft
    # https://work-in-progress.hatenablog.com/entry/2020/02/26/214211

    if False:
        frame_length=200
        frame_step=80
        fft_length=256
    else:
        # https://keras.io/examples/audio/ctc_asr/ values
        # An integer scalar Tensor. The window length in samples.
        frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        fft_length = 384    

    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    #stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x

def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return audio_ds

'''
def create_tf_dataset(data,bs)
    bs: batch size
'''
#def create_tf_dataset(data, bs=4):
def create_tf_dataset(data, vectorizer, bs=4):
    audio_ds = create_audio_ds(data)
    #text_ds = create_text_ds(data)
    text_ds = create_text_ds(data,vectorizer)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

"""
3. mltu class for mine 
"""
if True:
    import typing
    #from mltu.transformers import Transformer as TransformerX
    import mltu.transformers as trp
    class LabelIndexer_my(trp.Transformer):
        """Convert label to index by vocab
        
        Attributes:
            vocab (typing.List[str]): List of characters in vocab
            vocab : vectorizer.char_index
        """
        def __init__(
            self, 
            vocab: typing.List[str], char_index
            ) -> None:
            self.vocab = vocab
            self.char_index = char_index

        def __call_org__(self, data: np.ndarray, label: np.ndarray):
            #print('LabelIndexer_my():#7')
            return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

        def __call__(self, data: np.ndarray, label: np.ndarray):
            #print('LabelIndexer_my():#7')
            text="<"+label+">"
            ss=[]
            for l in text:
                if l in self.char_index:
                    n = self.char_index[l]
                    ss.append(n)
            return data, np.array(ss)

        def __call_test__(self, data: np.ndarray, label: np.ndarray):
            ss=[]
            #print('LabelIndexer_my():#7')
            for l in label:
                if l in self.char_index:
                    n = self.char_index[l]
                    ss.append(n)
            return data, np.array(ss)
            #return {"source":data, "target":np.array(ss)}
            #return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

    class SpectrogramPadding_my(trp.Transformer):
        """Pad spectrogram to max_spectrogram_length
        
        Attributes:
            max_spectrogram_length (int): Maximum length of spectrogram
            padding_value (int): Value to pad
        """
        def __init__(
            self, 
            max_spectrogram_length: int, 
            padding_value: int,
            append: bool = True
            ) -> None:
            self.max_spectrogram_length = max_spectrogram_length
            self.padding_value = padding_value
            self.append=append

        def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
            #print('spectrogram.shape:',spectrogram.shape)
            # spectrogram.shape: (1032, 193)
            if self.append==False:
                padded_spectrogram = np.pad(spectrogram, 
                    ((self.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)),mode="constant",constant_values=self.padding_value)
            else:
                l,h =spectrogram.shape
                lng = self.max_spectrogram_length - l
                if lng > 0:
                    a = np.full((lng,h),self.padding_value)
                    padded_spectrogram = np.append(spectrogram, a, axis=0)
                else:
                    padded_spectrogram = spectrogram
            return padded_spectrogram, label

    #  DataProvider for asr
    class DataProviderAsr(DataProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def __getitem__(self, index: int):
            #print('DataProviderAsr():#1')
            """ Returns a batch of data by batch index"""
            dataset_batch = self.get_batch_annotations(index)
        
            # First read and preprocess the batch data
            batch_data, batch_annotations = [], []
            for index, batch in enumerate(dataset_batch):

                data, annotation = self.process_data(batch)

                if data is None or annotation is None:
                    self.logger.warning("Data or annotation is None, skipping.")
                    continue

                batch_data.append(data)
                batch_annotations.append(annotation)

            so=np.array(batch_data)
            #print('so.shape:',so.shape)
            # so.shape: (8, 1392, 193)
            sa=np.array(batch_annotations)
            #print('sa.shape:',sa.shape)
            # sa.shape: (8, 189)
            
            #r=[so, sa]
            #print('type',type(r))
            #print('np.shape(r[0])',np.shape(r[0]))
            #print('np.shape(r[1])',np.shape(r[1]))
            #return [r]
            # change by nishi 2024.10.1
            return (so, sa)
