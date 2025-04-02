# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy

import utils.mel_features as mel_features
import utils.audio_params as audio_params
import pdb

def waveform_to_feature(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != audio_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, audio_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.

  '''这一步只是生成了梅尔频谱图而已'''
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=audio_params.SAMPLE_RATE,
      log_offset=audio_params.LOG_OFFSET,
      window_length_secs=audio_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=audio_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=audio_params.NUM_MEL_BINS,
      lower_edge_hertz=audio_params.MEL_MIN_HZ,
      upper_edge_hertz=audio_params.MEL_MAX_HZ)
  #print('-----打印一下log_mel的形状-----:', log_mel.shape)  # (2427, 64)


  # Frame features into examples.
  features_sample_rate = 1.0 / audio_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      audio_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))

  example_hop_length = int(round(
      audio_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  #print('-----打印一下example_window_length的值-----:', example_window_length)  # 64
  #print('-----打印一下example_hop_length的值-----:', example_hop_length)  # 2

  # (2427, 64)----->(1182, 64, 64)
  '''该函数将对数梅尔频谱图得到的n维数据，转换成了n+1维的'''
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,   # 每一帧的样本数
      hop_length=example_hop_length)         # 每个窗口之间的间隔
  #print('-----打印一下log_mel_examples的形状-----:', log_mel_examples.shape)  # (1182, 64, 64)
  return log_mel_examples


