# %%
# Melody_Harmonization_MAESTRO.ipynb

# Cell 1: Setup, Library Imports, and Global Configurations
import music21 as m21
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import time
from tqdm import tqdm
import pickle
from collections import Counter, defaultdict
import random
import json
from pathlib import Path
import logging

# Configure plots for better readability
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
FIG_SIZE = (12, 6)
SMALL_FIG_SIZE = (10, 4)

# --- Path Definitions ---
# Assumes the notebook is run from 'assignment2/CSE253R_A2/'
NOTEBOOK_CWD_ASSUMED = '.'

# MAESTRO_DATA_ROOT is the directory containing the year folders (2004, 2006, etc.)
# AND the 'maestro-v3.0.0' subdirectory (which contains the CSV).
MAESTRO_DATA_ROOT = os.path.join(NOTEBOOK_CWD_ASSUMED, 'data') # Resolves to './data/'

# Path to the directory containing maestro-v3.0.0.csv
METADATA_SUBDIR = 'maestro-v3.0.0'
METADATA_DIR_PATH = './data' # Resolves to './data/maestro-v3.0.0/'

METADATA_FILE = os.path.join(METADATA_DIR_PATH, 'maestro-v3.0.0.csv') # Correctly ./data/maestro-v3.0.0/maestro-v3.0.0.csv

# midi_features.json is at the root of CSE253R_A2 directory (notebook's CWD)
MIDI_FEATURES_PATH = os.path.join(NOTEBOOK_CWD_ASSUMED, 'midi_features.json') # Correctly ./midi_features.json

# Paths for saving/loading processed data for this harmonization task
PROCESSED_DATA_DIR = os.path.join(METADATA_DIR_PATH, 'processed_harmonization_data') # ./data/maestro-v3.0.0/processed_harmonization_data/
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

RAW_MELODY_HARMONY_PATH = os.path.join(PROCESSED_DATA_DIR, 'raw_melody_harmony_pairs.pkl')
VOCAB_PATH = os.path.join(PROCESSED_DATA_DIR, 'harmonization_vocab.pkl')
TOKENIZED_MELODY_HARMONY_PATH = os.path.join(PROCESSED_DATA_DIR, 'tokenized_melody_harmony_data.pkl')
PROCESSED_CONDITIONED_SEQUENCES_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_conditioned_sequences.pkl')
MODEL_SAVE_PATH_COND = os.path.join(PROCESSED_DATA_DIR, 'conditioned_melody_harmony_lstm.pth')

GENERATED_MIDI_DIR = 'generated_music_harmonization' # Relative to notebook CWD
os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)


# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print library versions
print(f"Music21 version: {m21.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Torch version: {torch.__version__}")

# --- Initial EDA on MAESTRO Metadata ---
metadata_df = None
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"\nSuccessfully loaded MAESTRO metadata from: {METADATA_FILE}")
    print("First 5 rows of the metadata:")
    print(metadata_df.head())
    print(f"\nShape of metadata: {metadata_df.shape}")
    print("\nColumns in metadata:")
    print(metadata_df.columns)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}")
    print(f"Please ensure your notebook's Current Working Directory is correct and the path is valid based on your directory structure.")
    print(f"Currently trying to access: {os.path.abspath(METADATA_FILE)}")
except Exception as e:
    print(f"An error occurred while loading the metadata: {e}")

if metadata_df is not None:
    print("\nDataset Split Counts:")
    split_counts = metadata_df['split'].value_counts()
    print(split_counts)

    print("\nTotal Duration (seconds) per Split:")
    duration_per_split = metadata_df.groupby('split')['duration'].sum()
    print(duration_per_split)
    total_duration_hours = metadata_df['duration'].sum() / 3600
    print(f"Total dataset duration: {total_duration_hours:.2f} hours")

    print("\nNumber of Unique Composers (Overall):")
    unique_composers_total = metadata_df['canonical_composer'].nunique()
    print(f"Total unique composers: {unique_composers_total}")

    # Plotting split distribution
    plt.figure(figsize=(10, 6))
    plot_order = ['train', 'validation', 'test']
    plot_order_present_in_data = [s for s in plot_order if s in split_counts.index]

    if plot_order_present_in_data:
        sns.countplot(data=metadata_df, x='split', order=plot_order_present_in_data, palette="viridis")
        plt.title('Distribution of MIDI Files Across Splits (MAESTRO v3.0.0)')
        plt.ylabel('Number of Files')
        plt.xlabel('Dataset Split')
        counts_for_plot = split_counts.reindex(plot_order_present_in_data).fillna(0)
        for i, split_name in enumerate(plot_order_present_in_data):
            count_val = counts_for_plot[split_name]
            plt.text(i, count_val + (0.01 * counts_for_plot.max()), str(int(count_val)), ha='center', va='bottom')
        plt.show()

        plt.figure(figsize=(10, 6))
        duration_per_split.reindex(plot_order_present_in_data).fillna(0).plot(
            kind='bar', 
            color=['#440154', '#21908d', '#fde725'][:len(plot_order_present_in_data)]
        )
        plt.title('Total Duration of Music Across Splits (MAESTRO v3.0.0)')
        plt.ylabel('Total Duration (seconds)')
        plt.xlabel('Dataset Split')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("No data splits found to plot distributions.")

# --- EDA with midi_features.json (if available) ---
midi_analysis_data = None
try:
    with open(MIDI_FEATURES_PATH, 'r') as f:
        midi_analysis_data = json.load(f)
    print(f"\nSuccessfully loaded {MIDI_FEATURES_PATH}.")
except FileNotFoundError:
    print(f"\nNote: {MIDI_FEATURES_PATH} not found. Skipping EDA based on these pre-extracted features.")
    print(f"Currently trying to access: {os.path.abspath(MIDI_FEATURES_PATH)}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {MIDI_FEATURES_PATH}.")

if midi_analysis_data and metadata_df is not None:
    features_df_from_json = pd.DataFrame(midi_analysis_data['features'])
    print("\nFirst 5 rows of the midi_features.json DataFrame:")
    print(features_df_from_json.head())
    print(f"\nShape of features_df_from_json: {features_df_from_json.shape}")

    print("\nOverall Statistics from midi_features.json:")
    for key, value in midi_analysis_data['statistics'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value:.2f}" if isinstance(sub_value, float) else f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    if 'short_midi_filename' not in metadata_df.columns:
        metadata_df['short_midi_filename'] = metadata_df['midi_filename'].apply(lambda x: os.path.basename(x))

    merged_features_df = pd.merge(features_df_from_json, metadata_df[['short_midi_filename', 'split', 'canonical_composer']],
                                  left_on='file_name', right_on='short_midi_filename', how='left')

    if not merged_features_df.empty and 'split' in merged_features_df.columns and plot_order_present_in_data:
        features_to_plot_eda = ['note_density', 'polyphony', 'duration', 'num_notes', 'mean_pitch', 'mean_velocity', 'mean_tempo']
        for feature in features_to_plot_eda:
            if feature in merged_features_df.columns:
                plt.figure(figsize=FIG_SIZE)
                sns.boxplot(data=merged_features_df, x='split', y=feature, order=plot_order_present_in_data, palette="viridis")
                plt.title(f'{feature.replace("_", " ").title()} Distribution per Split (from midi_features.json)')
                plt.ylabel(feature.replace("_", " ").title())
                plt.xlabel('Dataset Split')

                valid_feature_data = merged_features_df[feature].dropna()
                if not valid_feature_data.empty and valid_feature_data.quantile(0.99) > 0 : 
                    plt.ylim(0, valid_feature_data.quantile(0.99) * 1.1) 
                plt.show()
    else:
        print("Could not generate plots for features per split from midi_features.json (check data and split column).")
else:
    if metadata_df is None:
        print("\nMetadata not loaded. Skipping all EDA.")

# %% [markdown]
# # Cell 2: Symbolic Representation Definition
# Symbolic Representation for Melody and Harmony
# For our melody harmonization task, we will represent musical events as a tuple: (pitch, duration_quantized, ioi_quantized, velocity).
# 
# Pitch: MIDI note number (integer from 0-127).
# Duration (Quantized): The length of a note in terms of quarter lengths. This value will be quantized to the nearest time_quantization_step (e.g., 0.125 for 32nd notes) to create a discrete vocabulary.
# Inter-Onset Interval (IOI) (Quantized): The time difference between the start of the current event and the start of the previous event within its own stream (i.e., melody IOIs are relative to previous melody event, bass IOIs to previous bass event). This will also be quantized. For the first event in a stream, IOI can represent the offset from the beginning of the piece/segment.
# Velocity: MIDI velocity (integer from 0-127), representing the loudness of a note. This may also be quantized into bins for a smaller vocabulary if needed, but for initial processing, we will keep the raw values.
# This representation will be used for both the melody (condition) and the bass/harmony (target) sequences.

# %%
# Cell 3: Melody and Harmony Extraction Functions
from collections import defaultdict # Ensure defaultdict is imported

# Define Time Quantization Step
TIME_QUANTIZATION_STEP = 0.125 # e.g., 32nd note if a quarter note is 1.0

def get_instrument_part_for_piano(score):
    """
    Attempts to find the most prominent piano part.
    If multiple piano parts, it might require more sophisticated logic.
    For MAESTRO, piano is usually the main instrument.
    """
    # Try to find a piano part first
    parts = score.getElementsByClass(m21.stream.Part)
    piano_parts = []
    for p in parts:
        instr = p.getInstrument()
        if instr and 'piano' in instr.instrumentName.lower():
            piano_parts.append(p)
    
    if piano_parts:
        # If multiple piano parts, maybe combine them or pick the one with most notes
        # For now, pick the first one found or the one with most notes
        if len(piano_parts) > 1:
            piano_parts.sort(key=lambda p: len(p.flat.notesAndRests), reverse=True)
        return piano_parts[0].flat.notesAndRests # Return flattened notes and rests
        
    # If no explicit piano part, check if the score itself contains notes (e.g., Type 0 MIDI)
    if not parts and score.flat.notesAndRests:
        return score.flat.notesAndRests
        
    # If parts exist but no piano, try the part with most notes
    if parts:
        parts.sort(key=lambda p: len(p.flat.notesAndRests), reverse=True)
        if parts[0].flat.notesAndRests:
            return parts[0].flat.notesAndRests
            
    return score.flat.notesAndRests # Fallback

def extract_melody_and_harmony_parts(midi_path, time_quantization_step=0.125):
    """
    Extracts highest (melody) and lowest (bass/harmony) lines from a MIDI file.
    Assumes piano performance where melody is often highest and bass is lowest.
    Returns two lists of (pitch, duration, ioi, velocity) event tuples.
    """
    melody_events_raw = []
    harmony_events_raw = [] # Represents the bass line

    try:
        score = m21.converter.parse(midi_path)
        # For MAESTRO, which is primarily piano, flattening often works well.
        # If there are distinct parts, get_instrument_part_for_piano can be used.
        # notes_and_rests = get_instrument_part_for_piano(score)
        # For simplicity and robustness with MAESTRO, let's use .chordify() then iterate measures.
        # This helps group notes by onsets more reliably for polyphonic music.
        
        chordified_score = score.chordify()
        
        last_melody_onset_ql = 0.0
        last_harmony_onset_ql = 0.0

        # Iterate through elements in the chordified score (mostly chords now, or rests)
        for element in chordified_score.flat.notesAndRests:
            current_onset_ql = float(element.offset)
            
            # Quantize duration
            element_duration_ql = float(element.duration.quarterLength)
            quantized_duration = round(element_duration_ql / time_quantization_step) * time_quantization_step
            
            if quantized_duration <= 0:
                continue

            if isinstance(element, m21.note.Note): # Single note after chordify (rare but possible)
                pitch = element.pitch.midi
                velocity = element.volume.velocity if element.volume is not None else 64

                # Add to both melody and harmony if it's a single line
                melody_ioi_ql = round((current_onset_ql - last_melody_onset_ql) / time_quantization_step) * time_quantization_step
                melody_events_raw.append((pitch, quantized_duration, melody_ioi_ql, velocity))
                last_melody_onset_ql = current_onset_ql

                harmony_ioi_ql = round((current_onset_ql - last_harmony_onset_ql) / time_quantization_step) * time_quantization_step
                harmony_events_raw.append((pitch, quantized_duration, harmony_ioi_ql, velocity)) # Using same note for bass
                last_harmony_onset_ql = current_onset_ql

            elif isinstance(element, m21.chord.Chord):
                if not element.pitches: continue # Skip empty chords

                # Melody: highest pitch
                melody_pitch = element.sortAscending().pitches[-1].midi
                # Use velocity of the highest note if possible, or chord's general velocity
                # For simplicity, use chord's general velocity or default
                melody_velocity = element.volume.velocity if element.volume is not None else 64
                
                melody_ioi_ql = round((current_onset_ql - last_melody_onset_ql) / time_quantization_step) * time_quantization_step
                melody_events_raw.append((melody_pitch, quantized_duration, melody_ioi_ql, melody_velocity))
                last_melody_onset_ql = current_onset_ql

                # Harmony/Bass: lowest pitch
                harmony_pitch = element.sortAscending().pitches[0].midi
                harmony_velocity = element.volume.velocity if element.volume is not None else 64 # Use chord's velocity
                
                harmony_ioi_ql = round((current_onset_ql - last_harmony_onset_ql) / time_quantization_step) * time_quantization_step
                harmony_events_raw.append((harmony_pitch, quantized_duration, harmony_ioi_ql, harmony_velocity))
                last_harmony_onset_ql = current_onset_ql
            
            # Rests are implicitly handled by the IOI of the next note/chord.
            # If explicit rests are needed, they would be added here with a special pitch token.

    except Exception as e:
        print(f"Error processing {midi_path} for melody/harmony: {e}")
        return None, None
        
    return melody_events_raw, harmony_events_raw

# Test with a sample MIDI file (if metadata_df is loaded)
if metadata_df is not None and not metadata_df.empty:
    sample_midi_relative_path = metadata_df['midi_filename'].iloc[0]
    # MAESTRO_DATA_ROOT should be the directory containing year folders (2004, etc.)
    sample_midi_full_path = os.path.join(MAESTRO_DATA_ROOT, sample_midi_relative_path)

    if os.path.exists(sample_midi_full_path):
        print(f"\nTesting melody/harmony extraction with sample MIDI: {sample_midi_full_path}")
        melody_raw, harmony_raw = extract_melody_and_harmony_parts(sample_midi_full_path, TIME_QUANTIZATION_STEP)
        if melody_raw and harmony_raw:
            print(f"  Extracted Melody (first 5 events): {melody_raw[:5]}")
            print(f"  Extracted Harmony/Bass (first 5 events): {harmony_raw[:5]}")
            print(f"  Total melody events: {len(melody_raw)}, Total harmony events: {len(harmony_raw)}")
        else:
            print("  Failed to extract melody/harmony from sample MIDI.")
    else:
        print(f"Sample MIDI file not found: {sample_midi_full_path}")
        print(f"Ensure MAESTRO_DATA_ROOT ('{MAESTRO_DATA_ROOT}') is set correctly.")
else:
    print("\nmetadata_df not loaded. Cannot test melody/harmony extraction.")

# %%
# Cell 4: Process All MAESTRO MIDI Files for Melody-Harmony Pairs & Save/Load

raw_melody_harmony_data = {'train': [], 'validation': [], 'test': []}

if os.path.exists(RAW_MELODY_HARMONY_PATH):
    print(f"Loading raw melody-harmony pairs from {RAW_MELODY_HARMONY_PATH}...")
    with open(RAW_MELODY_HARMONY_PATH, 'rb') as f:
        raw_melody_harmony_data = pickle.load(f)
    print("Loaded successfully.")
else:
    if metadata_df is not None:
        print(f"Raw melody-harmony pairs not found. Processing from MIDI files...")
        # MAESTRO_DATA_ROOT is the root of the dataset (e.g., contains '2004/', '2006/' folders)
        
        for split in ['train', 'validation', 'test']:
            print(f"Processing {split} set...")
            split_df = metadata_df[metadata_df['split'] == split]
            for index, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Extracting {split} melody/harmony"):
                relative_midi_path = row['midi_filename']
                # The relative_midi_path in MAESTRO CSV is like '2004/MIDI-....midi'
                # MAESTRO_DATA_ROOT should be the path to the directory containing these '2004', '2006', etc. folders.
                full_midi_path = os.path.join(MAESTRO_DATA_ROOT, relative_midi_path)

                if os.path.exists(full_midi_path):
                    melody_seq, harmony_seq = extract_melody_and_harmony_parts(full_midi_path, TIME_QUANTIZATION_STEP)
                    if melody_seq and harmony_seq and len(melody_seq) > 1 and len(harmony_seq) > 1 : # Ensure non-empty sequences
                        raw_melody_harmony_data[split].append({'melody': melody_seq, 'harmony': harmony_seq, 'file': relative_midi_path})
                else:
                    print(f"Warning: MIDI file not found at {full_midi_path}")
        
        with open(RAW_MELODY_HARMONY_PATH, 'wb') as f:
            pickle.dump(raw_melody_harmony_data, f)
        print(f"Raw melody-harmony pairs saved to {RAW_MELODY_HARMONY_PATH}")
    else:
        print("metadata_df not loaded. Cannot process MIDI files.")

# Print statistics
if raw_melody_harmony_data:
    for split in ['train', 'validation', 'test']:
        num_pairs = len(raw_melody_harmony_data[split])
        print(f"Number of melody-harmony pairs in {split}: {num_pairs}")
        if num_pairs > 0:
            # Check lengths of first pair
            first_pair = raw_melody_harmony_data[split][0]
            print(f"  Example first pair in {split}:")
            print(f"    Melody len: {len(first_pair['melody'])}, First 3 melody events: {first_pair['melody'][:3]}")
            print(f"    Harmony len: {len(first_pair['harmony'])}, First 3 harmony events: {first_pair['harmony'][:3]}")
else:
    print("Failed to load or process raw melody-harmony pairs.")

# %%
# Cell 5: Vocabulary and Tokenization for Melody-Harmony Data

# Define special tokens (integer values, typically 0 for padding)
PAD_TOKEN_VALUE = 0
# UNK_TOKEN_VALUE = 1 # If you want explicit unknown tokens
# SOS_TOKEN_VALUE = 2 # Start of Sequence
# EOS_TOKEN_VALUE = 3 # End of Sequence
# For this task, we'll primarily use PAD_TOKEN_VALUE for padding.
# SOS/EOS can be handled by model architecture or prepending/appending to sequences if needed.

# Initialize vocabularies
pitch_to_int = {'<PAD_PITCH>': PAD_TOKEN_VALUE}
duration_to_int = {'<PAD_DUR>': PAD_TOKEN_VALUE}
ioi_to_int = {'<PAD_IOI>': PAD_TOKEN_VALUE}
velocity_to_int = {'<PAD_VEL>': PAD_TOKEN_VALUE}

int_to_pitch, int_to_duration, int_to_ioi, int_to_velocity = {}, {}, {}, {}

# --- Build or Load Vocabularies ---
if os.path.exists(VOCAB_PATH):
    print(f"Loading vocabularies from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)
    pitch_to_int = vocab_data['pitch_to_int']
    int_to_pitch = vocab_data['int_to_pitch']
    duration_to_int = vocab_data['duration_to_int']
    int_to_duration = vocab_data['int_to_duration']
    ioi_to_int = vocab_data['ioi_to_int']
    int_to_ioi = vocab_data['int_to_ioi']
    velocity_to_int = vocab_data['velocity_to_int']
    int_to_velocity = vocab_data['int_to_velocity']
    print("Vocabularies loaded.")
else:
    print("Vocabularies not found. Building from training data...")
    if raw_melody_harmony_data and raw_melody_harmony_data['train']:
        all_pitches_train = set()
        all_durations_train = set()
        all_iois_train = set()
        all_velocities_train = set()

        for pair in tqdm(raw_melody_harmony_data['train'], desc="Analyzing training data for vocabs"):
            for stream_type in ['melody', 'harmony']:
                for event in pair[stream_type]:
                    all_pitches_train.add(event[0])
                    all_durations_train.add(event[1])
                    all_iois_train.add(event[2])
                    all_velocities_train.add(event[3])
        
        # Start token assignment from 1 (0 is PAD)
        current_idx = 1
        for pitch in sorted(list(all_pitches_train)):
            if pitch not in pitch_to_int: pitch_to_int[pitch] = current_idx; current_idx +=1
        pitch_vocab_size = len(pitch_to_int)
        int_to_pitch = {i: p for p, i in pitch_to_int.items()}
        print(f"Pitch vocabulary size: {pitch_vocab_size}")

        current_idx = 1
        for dur in sorted(list(all_durations_train)):
            if dur not in duration_to_int: duration_to_int[dur] = current_idx; current_idx+=1
        duration_vocab_size = len(duration_to_int)
        int_to_duration = {i: d for d, i in duration_to_int.items()}
        print(f"Duration vocabulary size: {duration_vocab_size}")

        current_idx = 1
        for ioi in sorted(list(all_iois_train)):
            if ioi not in ioi_to_int: ioi_to_int[ioi] = current_idx; current_idx+=1
        ioi_vocab_size = len(ioi_to_int)
        int_to_ioi = {i: io for io, i in ioi_to_int.items()}
        print(f"IOI vocabulary size: {ioi_vocab_size}")
        
        # Discretize velocity into bins (e.g., 32 bins for 0-127 range)
        # Or use unique values if not too many. For simplicity, map unique values.
        current_idx = 1
        for vel in sorted(list(all_velocities_train)):
             if vel not in velocity_to_int: velocity_to_int[vel] = current_idx; current_idx+=1
        velocity_vocab_size = len(velocity_to_int)
        int_to_velocity = {i: v for v, i in velocity_to_int.items()}
        print(f"Velocity vocabulary size: {velocity_vocab_size}")

        vocab_data_to_save = {
            'pitch_to_int': pitch_to_int, 'int_to_pitch': int_to_pitch,
            'duration_to_int': duration_to_int, 'int_to_duration': int_to_duration,
            'ioi_to_int': ioi_to_int, 'int_to_ioi': int_to_ioi,
            'velocity_to_int': velocity_to_int, 'int_to_velocity': int_to_velocity
        }
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab_data_to_save, f)
        print(f"Vocabularies built and saved to {VOCAB_PATH}")
    else:
        print("Training data not available to build vocabularies. Please process data first.")
        # Define some default vocab sizes if training data is missing, for model instantiation later
        pitch_vocab_size = 128 
        duration_vocab_size = 50
        ioi_vocab_size = 50
        velocity_vocab_size = 32 # Example based on bins

# --- Tokenization Function ---
def tokenize_event_sequence(raw_event_sequence, pitch_map, duration_map, ioi_map, velocity_map):
    tokenized_events = []
    for pitch, dur, ioi, vel in raw_event_sequence:
        # Use PAD_TOKEN_VALUE (0) if token not found (though ideally all should be in vocab from training)
        p_tok = pitch_map.get(pitch, PAD_TOKEN_VALUE) 
        d_tok = duration_map.get(dur, PAD_TOKEN_VALUE)
        i_tok = ioi_map.get(ioi, PAD_TOKEN_VALUE)
        v_tok = velocity_map.get(vel, PAD_TOKEN_VALUE)
        tokenized_events.append((p_tok, d_tok, i_tok, v_tok))
    return tokenized_events

# --- Tokenize all sequences ---
tokenized_melody_harmony_data = {'train': [], 'validation': [], 'test': []}
if os.path.exists(TOKENIZED_MELODY_HARMONY_PATH):
    print(f"Loading tokenized melody-harmony data from {TOKENIZED_MELODY_HARMONY_PATH}...")
    with open(TOKENIZED_MELODY_HARMONY_PATH, 'rb') as f:
        tokenized_melody_harmony_data = pickle.load(f)
    print("Loaded successfully.")
elif raw_melody_harmony_data and pitch_to_int: # Check if raw data and vocabs exist
    print("Tokenizing melody-harmony sequences...")
    for split in ['train', 'validation', 'test']:
        for pair_dict in tqdm(raw_melody_harmony_data[split], desc=f"Tokenizing {split} pairs"):
            tokenized_melody = tokenize_event_sequence(pair_dict['melody'], pitch_to_int, duration_to_int, ioi_to_int, velocity_to_int)
            tokenized_harmony = tokenize_event_sequence(pair_dict['harmony'], pitch_to_int, duration_to_int, ioi_to_int, velocity_to_int)
            tokenized_melody_harmony_data[split].append({
                'melody': tokenized_melody,
                'harmony': tokenized_harmony,
                'file': pair_dict['file']
            })
    with open(TOKENIZED_MELODY_HARMONY_PATH, 'wb') as f:
        pickle.dump(tokenized_melody_harmony_data, f)
    print(f"Tokenized melody-harmony data saved to {TOKENIZED_MELODY_HARMONY_PATH}")
else:
    print("Raw data or vocabularies not available for tokenization.")

if tokenized_melody_harmony_data and tokenized_melody_harmony_data['train']:
    print("\nTokenization example (first training pair, first 3 events):")
    first_tokenized_pair = tokenized_melody_harmony_data['train'][0]
    print(f"  Melody (tokenized): {first_tokenized_pair['melody'][:3]}")
    print(f"  Harmony (tokenized): {first_tokenized_pair['harmony'][:3]}")

# PAD_TOKEN_IDs for embedding layers (should be 0 for all if PAD_TOKEN_VALUE is 0)
PAD_PITCH_TOKEN_ID = pitch_to_int.get('<PAD_PITCH>', 0)
PAD_DURATION_TOKEN_ID = duration_to_int.get('<PAD_DUR>', 0)
PAD_IOI_TOKEN_ID = ioi_to_int.get('<PAD_IOI>', 0)
PAD_VELOCITY_TOKEN_ID = velocity_to_int.get('<PAD_VEL>', 0)

# %%
# Cell 6: Sequence Padding and Truncation for Conditioned Data

MAX_SEQ_LENGTH = 256 # Define a suitable max sequence length (tune based on EDA of lengths)
PAD_EVENT_TOKEN_TUPLE = (PAD_PITCH_TOKEN_ID, PAD_DURATION_TOKEN_ID, PAD_IOI_TOKEN_ID, PAD_VELOCITY_TOKEN_ID)

# --- Determine MAX_SEQ_LENGTH from training data if not predefined ---
if tokenized_melody_harmony_data and tokenized_melody_harmony_data['train'] and MAX_SEQ_LENGTH is None:
    all_lengths = []
    for pair in tokenized_melody_harmony_data['train']:
        all_lengths.append(len(pair['melody']))
        all_lengths.append(len(pair['harmony']))
    if all_lengths:
        MAX_SEQ_LENGTH = int(np.percentile(all_lengths, 95))
        print(f"Calculated MAX_SEQ_LENGTH (95th percentile): {MAX_SEQ_LENGTH}")
    else:
        MAX_SEQ_LENGTH = 200 # Default if no data
        print(f"Using default MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH} due to no training data for length analysis.")
elif MAX_SEQ_LENGTH is None:
    MAX_SEQ_LENGTH = 200 # Default
    print(f"Using default MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
else:
    print(f"Using predefined MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")


def pad_or_truncate_tokenized_sequence(tokenized_sequence, max_len, pad_event_tuple):
    if len(tokenized_sequence) > max_len:
        return tokenized_sequence[:max_len]  # Truncate
    elif len(tokenized_sequence) < max_len:
        return tokenized_sequence + [pad_event_tuple] * (max_len - len(tokenized_sequence)) # Pad
    return tokenized_sequence

# --- Process and Save/Load Padded/Truncated Sequences ---
processed_conditioned_sequences = {'train': [], 'validation': [], 'test': []}

if os.path.exists(PROCESSED_CONDITIONED_SEQUENCES_PATH):
    print(f"Loading processed (padded/truncated) conditioned sequences from {PROCESSED_CONDITIONED_SEQUENCES_PATH}...")
    with open(PROCESSED_CONDITIONED_SEQUENCES_PATH, 'rb') as f:
        processed_conditioned_sequences = pickle.load(f)
    print("Loaded successfully.")
elif tokenized_melody_harmony_data and pitch_to_int: # Check if tokenized data and vocabs exist
    print("Padding/Truncating tokenized sequences...")
    for split in ['train', 'validation', 'test']:
        for pair_dict in tqdm(tokenized_melody_harmony_data[split], desc=f"Padding/Truncating {split} pairs"):
            
            padded_melody_tokens = pad_or_truncate_tokenized_sequence(pair_dict['melody'], MAX_SEQ_LENGTH, PAD_EVENT_TOKEN_TUPLE)
            padded_harmony_tokens = pad_or_truncate_tokenized_sequence(pair_dict['harmony'], MAX_SEQ_LENGTH, PAD_EVENT_TOKEN_TUPLE)
            
            # Store as dictionaries of component tensors for easier Dataset creation
            # Melody (Condition)
            cond_pitch = torch.tensor([e[0] for e in padded_melody_tokens], dtype=torch.long)
            cond_dur = torch.tensor([e[1] for e in padded_melody_tokens], dtype=torch.long)
            cond_ioi = torch.tensor([e[2] for e in padded_melody_tokens], dtype=torch.long)
            cond_vel = torch.tensor([e[3] for e in padded_melody_tokens], dtype=torch.long)
            
            # Harmony (Target)
            target_pitch = torch.tensor([e[0] for e in padded_harmony_tokens], dtype=torch.long)
            target_dur = torch.tensor([e[1] for e in padded_harmony_tokens], dtype=torch.long)
            target_ioi = torch.tensor([e[2] for e in padded_harmony_tokens], dtype=torch.long)
            target_vel = torch.tensor([e[3] for e in padded_harmony_tokens], dtype=torch.long)

            processed_conditioned_sequences[split].append({
                'condition': (cond_pitch, cond_dur, cond_ioi, cond_vel),
                'target': (target_pitch, target_dur, target_ioi, target_vel),
                'file': pair_dict['file']
            })
            
    with open(PROCESSED_CONDITIONED_SEQUENCES_PATH, 'wb') as f:
        pickle.dump(processed_conditioned_sequences, f)
    print(f"Processed (padded/truncated) conditioned sequences saved to {PROCESSED_CONDITIONED_SEQUENCES_PATH}")
else:
    print("Tokenized data not available for padding/truncation.")

if processed_conditioned_sequences and processed_conditioned_sequences['train']:
    print("\nPadding/Truncation example (first training pair):")
    first_proc_pair = processed_conditioned_sequences['train'][0]
    print(f"  Condition Pitch Tensor Shape: {first_proc_pair['condition'][0].shape}") # Should be (MAX_SEQ_LENGTH,)
    print(f"  Target Pitch Tensor Shape: {first_proc_pair['target'][0].shape}")   # Should be (MAX_SEQ_LENGTH,)
    print(f"  First 3 condition events (pitch tokens): {first_proc_pair['condition'][0][:3]}")
    print(f"  First 3 target events (pitch tokens): {first_proc_pair['target'][0][:3]}")

# %%
# Cell 7: Conditioned Music Dataset and DataLoader (PyTorch)

class ConditionedMusicDataset(Dataset):
    def __init__(self, processed_conditioned_split_data):
        # processed_conditioned_split_data is a list of dicts:
        # [{'condition': (cond_p_tensor, cond_d_tensor, cond_i_tensor, cond_v_tensor), 
        #   'target': (tar_p_tensor, tar_d_tensor, tar_i_tensor, tar_v_tensor)}, ...]
        # Each tensor is of shape (MAX_SEQ_LENGTH)
        self.data = processed_conditioned_split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        condition_tensors = item['condition'] # tuple of 4 tensors
        target_tensors = item['target']       # tuple of 4 tensors

        # For Seq2Seq, the decoder input at step t is usually target_event[t-1] (for teacher forcing)
        # and it predicts target_event[t].
        # The DataLoader will serve the full sequences; slicing for input/target
        # will be handled by the model's forward pass or in the training loop for clarity.
        # For now, we return the condition and the full target sequence.
        
        # Condition (Melody)
        cond_pitch = condition_tensors[0]
        cond_dur = condition_tensors[1]
        cond_ioi = condition_tensors[2]
        cond_vel = condition_tensors[3]

        # Target (Harmony/Bass)
        # The decoder will typically take target_X[:, :-1] as input and predict target_X[:, 1:]
        # Or, more explicitly, input target_X[t] to predict target_X[t+1]
        # For now, let's pass the full target sequences. The model's forward pass will handle the shifting.
        target_pitch = target_tensors[0]
        target_dur = target_tensors[1]
        target_ioi = target_tensors[2]
        target_vel = target_tensors[3]
        
        return {
            'cond_pitch': cond_pitch, 'cond_duration': cond_dur,
            'cond_ioi': cond_ioi, 'cond_velocity': cond_vel,
            'target_pitch': target_pitch, 'target_duration': target_dur,
            'target_ioi': target_ioi, 'target_velocity': target_vel
        }

# --- Create DataLoaders ---
train_dataloader_cond, validation_dataloader_cond, test_dataloader_cond = None, None, None
BATCH_SIZE = 32 # Tune this

if processed_conditioned_sequences and processed_conditioned_sequences['train']:
    train_dataset_cond = ConditionedMusicDataset(processed_conditioned_sequences['train'])
    train_dataloader_cond = DataLoader(train_dataset_cond, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    print(f"Created training DataLoader. Batches: {len(train_dataloader_cond)}")
    
    if processed_conditioned_sequences['validation']:
        validation_dataset_cond = ConditionedMusicDataset(processed_conditioned_sequences['validation'])
        validation_dataloader_cond = DataLoader(validation_dataset_cond, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
        print(f"Created validation DataLoader. Batches: {len(validation_dataloader_cond)}")
    else:
        print("No validation data for conditioned task.")

    if processed_conditioned_sequences['test']:
        test_dataset_cond = ConditionedMusicDataset(processed_conditioned_sequences['test'])
        test_dataloader_cond = DataLoader(test_dataset_cond, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
        print(f"Created test DataLoader. Batches: {len(test_dataloader_cond)}")
    else:
        print("No test data for conditioned task.")
        
    # Example: Fetch one batch to check shapes
    if train_dataloader_cond and len(train_dataloader_cond) > 0:
        sample_batch_cond = next(iter(train_dataloader_cond))
        print("\nSample conditioned batch shapes:")
        print(f"  Cond Pitch: {sample_batch_cond['cond_pitch'].shape}")     # Expected: (BATCH_SIZE, MAX_SEQ_LENGTH)
        print(f"  Target Pitch: {sample_batch_cond['target_pitch'].shape}") # Expected: (BATCH_SIZE, MAX_SEQ_LENGTH)
else:
    print("Processed conditioned sequences not available. Skipping DataLoader creation.")

# %%
# Cell 8: Seq2Seq Model Architecture (Encoder, Attention, Decoder, Seq2SeqMusic)

# Ensure PAD token IDs are defined (from Cell 5)
# If not, define defaults (though this means vocabs weren't built correctly)
PAD_PITCH_TOKEN_ID = pitch_to_int.get('<PAD_PITCH>', 0) if 'pitch_to_int' in locals() else 0
PAD_DURATION_TOKEN_ID = duration_to_int.get('<PAD_DUR>', 0) if 'duration_to_int' in locals() else 0
PAD_IOI_TOKEN_ID = ioi_to_int.get('<PAD_IOI>', 0) if 'ioi_to_int' in locals() else 0
PAD_VELOCITY_TOKEN_ID = velocity_to_int.get('<PAD_VEL>', 0) if 'velocity_to_int' in locals() else 0

# Ensure vocab sizes are defined (from Cell 5) or provide defaults
pitch_vocab_size = len(pitch_to_int) if 'pitch_to_int' in locals() and pitch_to_int else 130 # MIDI 0-127 + PAD + UNK
duration_vocab_size = len(duration_to_int) if 'duration_to_int' in locals() and duration_to_int else 50
ioi_vocab_size = len(ioi_to_int) if 'ioi_to_int' in locals() and ioi_to_int else 50
velocity_vocab_size = len(velocity_to_int) if 'velocity_to_int' in locals() and velocity_to_int else 32 # e.g. binned


class EncoderLSTM(nn.Module):
    def __init__(self, p_vocab, d_vocab, i_vocab, v_vocab,
                 emb_p, emb_d, emb_i, emb_v,
                 hidden_dim, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.pitch_embedding = nn.Embedding(p_vocab, emb_p, padding_idx=PAD_PITCH_TOKEN_ID)
        self.duration_embedding = nn.Embedding(d_vocab, emb_d, padding_idx=PAD_DURATION_TOKEN_ID)
        self.ioi_embedding = nn.Embedding(i_vocab, emb_i, padding_idx=PAD_IOI_TOKEN_ID)
        self.velocity_embedding = nn.Embedding(v_vocab, emb_v, padding_idx=PAD_VELOCITY_TOKEN_ID)
        
        combined_emb_dim = emb_p + emb_d + emb_i + emb_v
        self.lstm = nn.LSTM(combined_emb_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                            bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim) # To make hidden compatible with unidirectional decoder
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, p_seq, d_seq, i_seq, v_seq):
        p_emb = self.dropout_layer(self.pitch_embedding(p_seq))
        d_emb = self.dropout_layer(self.duration_embedding(d_seq))
        i_emb = self.dropout_layer(self.ioi_embedding(i_seq))
        v_emb = self.dropout_layer(self.velocity_embedding(v_seq))
        
        combined_embeds = torch.cat((p_emb, d_emb, i_emb, v_emb), dim=2)
        
        outputs, (hidden, cell) = self.lstm(combined_embeds)
        # outputs: (batch_size, seq_len, hidden_dim * 2)
        # hidden, cell: (num_layers * 2, batch_size, hidden_dim)

        # Combine bidirectional hidden states for decoder initialization
        # Reshape hidden and cell: (num_layers, 2 (directions), batch, hidden_dim)
        # Then take the last layer: hidden[-1] -> (2, batch, hidden_dim) -> cat -> (batch, hidden_dim*2)
        # We need (num_decoder_layers, batch_size, decoder_hidden_dim)
        # If decoder is unidirectional and has same hidden_dim, fc_hidden/cell are used.
        
        # Concatenate the hidden states from the forward and backward LSTM layers
        # hidden is (num_layers*2, batch, hidden_dim), we want (num_layers_decoder, batch, hidden_dim_decoder)
        # For a single layer bidirectional encoder and single layer unidirectional decoder:
        # hidden shape (2, batch, enc_hidden_dim) -> (batch, enc_hidden_dim * 2) -> fc -> (batch, dec_hidden_dim) -> unsqueeze(0)
        
        # Combine final forward and backward hidden states
        # hidden is (num_layers*num_directions, batch, hidden_size)
        # For single layer bidirectional: (2, batch, hidden_dim)
        # Concatenate the forward (hidden[0,:,:]) and backward (hidden[1,:,:]) hidden states
        h_n = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        c_n = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        
        # If decoder has multiple layers, repeat this combined state
        # For simplicity, assuming decoder num_layers matches encoder num_layers (unidirectional sense)
        # For this example, assume decoder.num_layers = encoder.num_layers (unidirectional)
        # So, if encoder is 1 layer bidirectional, and decoder is 1 layer unidirectional, h_n, c_n are (batch, hidden_dim)
        # We need to make them (decoder_num_layers, batch, hidden_dim)
        # If decoder is single layer:
        h_n_decoder = h_n.unsqueeze(0) # (1, batch, hidden_dim)
        c_n_decoder = c_n.unsqueeze(0) # (1, batch, hidden_dim)

        return outputs, (h_n_decoder, c_n_decoder)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # Encoder hidden dim is *2 if bidirectional
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: (batch_size, dec_hid_dim) - from decoder's previous step (top layer)
        # encoder_outputs shape: (batch_size, src_len, enc_hid_dim * 2)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2) # (batch_size, src_len)
        return F.softmax(attention, dim=1)


class DecoderLSTM(nn.Module):
    def __init__(self, p_vocab, d_vocab, i_vocab, v_vocab,
                 emb_p, emb_d, emb_i, emb_v,
                 enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()
        self.pitch_vocab_size = p_vocab
        self.duration_vocab_size = d_vocab
        self.ioi_vocab_size = i_vocab
        self.velocity_vocab_size = v_vocab
        self.attention = attention
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers

        self.pitch_embedding = nn.Embedding(p_vocab, emb_p, padding_idx=PAD_PITCH_TOKEN_ID)
        self.duration_embedding = nn.Embedding(d_vocab, emb_d, padding_idx=PAD_DURATION_TOKEN_ID)
        self.ioi_embedding = nn.Embedding(i_vocab, emb_i, padding_idx=PAD_IOI_TOKEN_ID)
        self.velocity_embedding = nn.Embedding(v_vocab, emb_v, padding_idx=PAD_VELOCITY_TOKEN_ID)
        
        combined_emb_dim = emb_p + emb_d + emb_i + emb_v
        # LSTM input: current embedded input + context vector (enc_hid_dim * 2 for bidir encoder)
        self.lstm = nn.LSTM(combined_emb_dim + (enc_hid_dim * 2), dec_hid_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc_pitch = nn.Linear(dec_hid_dim, p_vocab)
        self.fc_duration = nn.Linear(dec_hid_dim, d_vocab)
        self.fc_ioi = nn.Linear(dec_hid_dim, i_vocab)
        self.fc_velocity = nn.Linear(dec_hid_dim, v_vocab)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, p_in, d_in, i_in, v_in, hidden, cell, encoder_outputs):
        # *_in are (batch_size) - current token for this step
        # hidden, cell are (num_layers, batch_size, dec_hid_dim)
        # encoder_outputs are (batch_size, src_len, enc_hid_dim * 2)

        p_in = p_in.unsqueeze(1) # (batch_size, 1)
        d_in = d_in.unsqueeze(1)
        i_in = i_in.unsqueeze(1)
        v_in = v_in.unsqueeze(1)

        p_emb = self.dropout_layer(self.pitch_embedding(p_in)) # (batch_size, 1, emb_dim)
        d_emb = self.dropout_layer(self.duration_embedding(d_in))
        i_emb = self.dropout_layer(self.ioi_embedding(i_in))
        v_emb = self.dropout_layer(self.velocity_embedding(v_in))
        
        embedded = torch.cat((p_emb, d_emb, i_emb, v_emb), dim=2) # (batch_size, 1, combined_emb_dim)
        
        # Attention: use top layer hidden state of decoder
        # hidden is (num_layers, batch, dec_hid_dim), so hidden[-1] is (batch, dec_hid_dim)
        attn_weights = self.attention(hidden[-1], encoder_outputs) # (batch_size, src_len)
        attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1, src_len)
        
        context_vector = torch.bmm(attn_weights, encoder_outputs) # (batch_size, 1, enc_hid_dim * 2)
        
        lstm_input = torch.cat((embedded, context_vector), dim=2) # (batch, 1, emb + enc_hid*2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, dec_hid_dim)
        # hidden, cell: (num_layers, batch_size, dec_hid_dim)
        
        output = output.squeeze(1) # (batch_size, dec_hid_dim)
        
        pred_p = self.fc_pitch(output)
        pred_d = self.fc_duration(output)
        pred_i = self.fc_ioi(output)
        pred_v = self.fc_velocity(output)
        
        return pred_p, pred_d, pred_i, pred_v, hidden, cell, attn_weights.squeeze(1)


class Seq2SeqMusic(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, cond_p, cond_d, cond_i, cond_v, 
                target_p, target_d, target_i, target_v, 
                teacher_forcing_ratio=0.5):
        # cond_* : (batch_size, cond_seq_len)
        # target_* : (batch_size, target_seq_len) - these are the ground truth for teacher forcing
        
        batch_size = target_p.shape[0]
        target_len = target_p.shape[1] # Max length of harmony to generate/predict

        outputs_p = torch.zeros(batch_size, target_len, self.decoder.pitch_vocab_size).to(self.device)
        outputs_d = torch.zeros(batch_size, target_len, self.decoder.duration_vocab_size).to(self.device)
        outputs_i = torch.zeros(batch_size, target_len, self.decoder.ioi_vocab_size).to(self.device)
        outputs_v = torch.zeros(batch_size, target_len, self.decoder.velocity_vocab_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(cond_p, cond_d, cond_i, cond_v)
        
        # First input to decoder is the <SOS> token or the first actual token of the target sequence.
        # For training, target_* are the full target sequences.
        # Decoder input at step t=0 (to predict target at t=0)
        # We'll use the actual first token of the target sequence (ground truth) as the initial input.
        # Or, if you have a specific <SOS> token, it would be that.
        # For simplicity, let's assume target_X[:,0] is the first event to be predicted (based on <SOS> or similar)
        # and the input to the decoder at step 0 is an <SOS> token for all components.
        # Let's make the decoder predict target_X[:, t] using target_X[:, t-1] as input (or <SOS> if t=0)
        
        # Initial decoder input (e.g., PAD tokens if no specific SOS used during training)
        # For generation, this will be a specific <SOS> or learned start token.
        # For training, we feed the first element of the target sequence (target_X[:, 0])
        # which means our model is predicting from the second element onwards based on the first.
        # Or, more standardly, the decoder inputs are target_X[:, :-1] and outputs are compared against target_X[:, 1:]
        # Let's adjust for standard Seq2Seq:
        # Input to decoder: target_X[:, t]
        # Prediction: for target_X[:, t+1]
        # The target_X passed here has length target_len. The loop runs target_len times.
        # The model predicts T outputs. These correspond to predicting target_X[0]...target_X[T-1].

        # For the first time step, input is <SOS> equivalent. We'll use PAD tokens as <SOS> for simplicity here.
        # A better approach would be dedicated <SOS> tokens.
        current_p_in = torch.full((batch_size,), PAD_PITCH_TOKEN_ID, dtype=torch.long).to(self.device)
        current_d_in = torch.full((batch_size,), PAD_DURATION_TOKEN_ID, dtype=torch.long).to(self.device)
        current_i_in = torch.full((batch_size,), PAD_IOI_TOKEN_ID, dtype=torch.long).to(self.device) # IOI for first generated can be 0
        current_v_in = torch.full((batch_size,), PAD_VELOCITY_TOKEN_ID, dtype=torch.long).to(self.device)

        for t in range(target_len): # Generate one event at a time
            pred_p, pred_d, pred_i, pred_v, hidden, cell, _ = self.decoder(
                current_p_in, current_d_in, current_i_in, current_v_in,
                hidden, cell, encoder_outputs
            )
            
            outputs_p[:, t, :] = pred_p
            outputs_d[:, t, :] = pred_d
            outputs_i[:, t, :] = pred_i
            outputs_v[:, t, :] = pred_v
            
            use_teacher_force = random.random() < teacher_forcing_ratio
            
            if use_teacher_force: # Next input is ground truth
                current_p_in = target_p[:, t]
                current_d_in = target_d[:, t]
                current_i_in = target_i[:, t]
                current_v_in = target_vel[:, t]
            else: # Next input is model's own prediction
                current_p_in = pred_p.argmax(1)
                current_d_in = pred_d.argmax(1)
                current_i_in = pred_i.argmax(1)
                current_v_in = pred_v.argmax(1)
                
        return outputs_p, outputs_d, outputs_i, outputs_v

# Hyperparameters
COND_EMB_PITCH = 64
COND_EMB_DURATION = 32
COND_EMB_IOI = 32
COND_EMB_VELOCITY = 32
COND_ENC_HID_DIM = 128 # For bidirectional LSTM, actual output dim will be *2
COND_DEC_HID_DIM = 256 # Often same as enc_hid_dim * 2 (if encoder is bidir) or just enc_hid_dim
COND_ENC_LAYERS = 1 # Bidirectional encoder
COND_DEC_LAYERS = 1 # Unidirectional decoder
COND_DROPOUT = 0.3

# Ensure device is defined
if 'device' not in locals(): device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention_module = Attention(COND_ENC_HID_DIM, COND_DEC_HID_DIM) # enc_hid_dim * 2 for bidir, dec_hid_dim
encoder = EncoderLSTM(pitch_vocab_size, duration_vocab_size, ioi_vocab_size, velocity_vocab_size,
                      COND_EMB_PITCH, COND_EMB_DURATION, COND_EMB_IOI, COND_EMB_VELOCITY,
                      COND_ENC_HID_DIM, COND_ENC_LAYERS, COND_DROPOUT)

decoder = DecoderLSTM(pitch_vocab_size, duration_vocab_size, ioi_vocab_size, velocity_vocab_size,
                      COND_EMB_PITCH, COND_EMB_DURATION, COND_EMB_IOI, COND_EMB_VELOCITY,
                      COND_ENC_HID_DIM, COND_DEC_HID_DIM, COND_DEC_LAYERS, COND_DROPOUT, attention_module)

cond_model = Seq2SeqMusic(encoder, decoder, device).to(device)
print(f"Conditioned Seq2Seq model created and moved to {device}.")
# print(cond_model) # Can be very long to print
num_params_cond = sum(p.numel() for p in cond_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in conditioned model: {num_params_cond:,}")

# %%
# Cell 9: Conditioned Model Training Loop

def tf_ratio_schedule_linear(epoch, total_epochs, start_ratio=1.0, end_ratio=0.0):
    """Linearly decays teacher forcing ratio."""
    if total_epochs <= 1: return end_ratio
    return max(end_ratio, start_ratio - (start_ratio - end_ratio) * (epoch / (total_epochs -1)))

def train_conditioned_model(model, train_loader, val_loader, num_epochs, learning_rate, device, 
                            model_save_path, tf_schedule_func=None, total_tf_epochs=None):
    criterion_pitch = nn.CrossEntropyLoss(ignore_index=PAD_PITCH_TOKEN_ID)
    criterion_duration = nn.CrossEntropyLoss(ignore_index=PAD_DURATION_TOKEN_ID)
    criterion_ioi = nn.CrossEntropyLoss(ignore_index=PAD_IOI_TOKEN_ID)
    criterion_velocity = nn.CrossEntropyLoss(ignore_index=PAD_VELOCITY_TOKEN_ID)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    if total_tf_epochs is None:
        total_tf_epochs = num_epochs # Default to decay over all epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()
        
        current_tf_ratio = tf_schedule_func(epoch, total_tf_epochs) if tf_schedule_func else 0.5
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} TF={current_tf_ratio:.2f} - Training")
        for batch in progress_bar:
            cond_p = batch['cond_pitch'].to(device)
            cond_d = batch['cond_duration'].to(device)
            cond_i = batch['cond_ioi'].to(device)
            cond_v = batch['cond_velocity'].to(device)
            
            target_p = batch['target_pitch'].to(device)
            target_d = batch['target_duration'].to(device)
            target_i = batch['target_ioi'].to(device)
            target_v = batch['target_velocity'].to(device)
            
            optimizer.zero_grad()
            
            # model's forward pass expects full target sequences for teacher forcing logic
            # The targets for loss are the same sequences, as the model's output
            # will align with predicting target[t] given input up to target[t-1] (or sos).
            out_p, out_d, out_i, out_v = model(cond_p, cond_d, cond_i, cond_v,
                                               target_p, target_d, target_i, target_v,
                                               teacher_forcing_ratio=current_tf_ratio)
            
            # Reshape for CrossEntropyLoss: (N * seq_len, C) and (N * seq_len)
            loss_p = criterion_pitch(out_p.reshape(-1, out_p.shape[-1]), target_p.reshape(-1))
            loss_d = criterion_duration(out_d.reshape(-1, out_d.shape[-1]), target_d.reshape(-1))
            loss_i = criterion_ioi(out_i.reshape(-1, out_i.shape[-1]), target_i.reshape(-1))
            loss_v = criterion_velocity(out_v.reshape(-1, out_v.shape[-1]), target_v.reshape(-1))
            
            # Combine losses (equal weighting for now)
            total_loss = loss_p + loss_d + loss_i + loss_v
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            progress_bar.set_postfix({'batch_loss': total_loss.item()})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                cond_p = batch['cond_pitch'].to(device)
                # ... (similar for other cond inputs)
                cond_d = batch['cond_duration'].to(device)
                cond_i = batch['cond_ioi'].to(device)
                cond_v = batch['cond_velocity'].to(device)

                target_p = batch['target_pitch'].to(device)
                # ... (similar for other target inputs)
                target_d = batch['target_duration'].to(device)
                target_i = batch['target_ioi'].to(device)
                target_v = batch['target_velocity'].to(device)
                
                # During validation, teacher forcing is off (ratio = 0.0)
                # The model will still need the target sequence start for the first input to the decoder
                # and the full target sequence length to know how many steps to generate.
                out_p, out_d, out_i, out_v = model(cond_p, cond_d, cond_i, cond_v,
                                                   target_p, target_d, target_i, target_v, 
                                                   teacher_forcing_ratio=0.0)

                loss_p = criterion_pitch(out_p.reshape(-1, out_p.shape[-1]), target_p.reshape(-1))
                loss_d = criterion_duration(out_d.reshape(-1, out_d.shape[-1]), target_d.reshape(-1))
                loss_i = criterion_ioi(out_i.reshape(-1, out_i.shape[-1]), target_i.reshape(-1))
                loss_v = criterion_velocity(out_v.reshape(-1, out_v.shape[-1]), target_v.reshape(-1))
                
                total_loss = loss_p + loss_d + loss_i + loss_v
                epoch_val_loss += total_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        history['val_loss'].append(avg_val_loss)
        
        epoch_duration_s = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_duration_s:.2f}s - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saved model to {model_save_path}")
            
    return history

# --- Start Training ---
NUM_EPOCHS_COND = 20 # Adjust as needed (e.g., 50-100 for real training)
LEARNING_RATE_COND = 0.001

if train_dataloader_cond and validation_dataloader_cond and cond_model:
    print("\nStarting training for conditioned Seq2Seq model...")
    
    # Define a teacher forcing schedule function that decays over NUM_EPOCHS_COND
    tf_scheduler = lambda epoch: tf_ratio_schedule_linear(epoch, NUM_EPOCHS_COND, start_ratio=1.0, end_ratio=0.05)

    cond_history = train_conditioned_model(
        cond_model, 
        train_dataloader_cond, 
        validation_dataloader_cond, 
        NUM_EPOCHS_COND, 
        LEARNING_RATE_COND, 
        device,
        MODEL_SAVE_PATH_COND,
        tf_schedule_func=tf_scheduler,
        total_tf_epochs=NUM_EPOCHS_COND 
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(cond_history['train_loss'], label='Train Loss')
    plt.plot(cond_history['val_loss'], label='Validation Loss')
    plt.title('Conditioned Seq2Seq Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(GENERATED_MIDI_DIR, "conditioned_training_loss.png"))
    plt.show()
else:
    print("Conditioned DataLoaders or model not available. Skipping conditioned model training.")

# %%
# Cell 10: Symbolic to MIDI Conversion Functions

def symbolic_events_to_midi_stream(symbolic_event_tuples, 
                                   int_to_pitch_map, int_to_duration_map, 
                                   int_to_ioi_map, int_to_velocity_map,
                                   stream_instrument=None):
    """
    Converts a list of symbolic event tuples to a music21 stream.Part.
    Each event tuple: (pitch_token, dur_token, ioi_token, vel_token)
    """
    part = m21.stream.Part()
    if stream_instrument:
        part.append(stream_instrument)
    else:
        part.append(m21.instrument.Piano()) # Default to piano

    current_offset_ql = 0.0
    first_event = True

    for p_tok, d_tok, i_tok, v_tok in symbolic_event_tuples:
        # Skip complete padding events
        if (p_tok == PAD_PITCH_TOKEN_ID and d_tok == PAD_DURATION_TOKEN_ID and \
            i_tok == PAD_IOI_TOKEN_ID and v_tok == PAD_VELOCITY_TOKEN_ID):
            continue

        pitch_val = int_to_pitch_map.get(p_tok)
        duration_val = int_to_duration_map.get(d_tok)
        ioi_val = int_to_ioi_map.get(i_tok)
        velocity_val = int_to_velocity_map.get(v_tok)

        # Handle potential None from map.get if a token is truly unknown (should be rare if vocab is complete)
        # or if a PAD token for a specific component was generated (which we should skip)
        if pitch_val is None or duration_val is None or ioi_val is None or velocity_val is None:
            print(f"Warning: Skipping event with None component after de-tokenization: {(p_tok, d_tok, i_tok, v_tok)}")
            continue
        if isinstance(pitch_val, str) and '<PAD' in pitch_val: continue
        if isinstance(duration_val, str) and '<PAD' in duration_val: continue
        # IOI can be 0, which is valid.
        if isinstance(ioi_val, str) and '<PAD' in ioi_val: continue # Should only happen if PAD_IOI is not 0 and it's generated
        if isinstance(velocity_val, str) and '<PAD' in velocity_val: continue
        
        # If duration or velocity became 0 due to PAD and were not caught above
        if duration_val == 0 or velocity_val == 0: # Assuming 0 velocity also means skip
             # Advance time by IOI even if note is skipped (e.g. represents a rest)
            if first_event:
                current_offset_ql = float(ioi_val) 
                first_event = False
            else:
                current_offset_ql += float(ioi_val)
            continue


        # IOI is time from start of previous event to start of current event
        if first_event:
            current_offset_ql = float(ioi_val) # For the first event, IOI is from time 0
            first_event = False
        else:
            current_offset_ql += float(ioi_val) # Advance by IOI from previous event's onset

        if 0 <= pitch_val <= 127: # Valid MIDI pitch
            note = m21.note.Note(pitch_val)
            note.duration = m21.duration.Duration(float(duration_val))
            note.volume.velocity = int(velocity_val)
            part.insert(current_offset_ql, note)
        elif pitch_val == -1: # Example: explicit rest token
            rest = m21.note.Rest()
            rest.duration = m21.duration.Duration(float(duration_val))
            part.insert(current_offset_ql, rest)
        # Else: could be another special token, or an invalid pitch if model generates out of range.
        
    return part

def symbolic_to_midi_file_combined(melody_tuples, harmony_tuples, output_midi_path,
                                   int_to_pitch_map, int_to_duration_map,
                                   int_to_ioi_map, int_to_velocity_map):
    """
    Converts tokenized melody and harmony sequences to a two-track MIDI file.
    """
    score = m21.stream.Score()
    
    melody_part = symbolic_events_to_midi_stream(melody_tuples, 
                                                 int_to_pitch_map, int_to_duration_map,
                                                 int_to_ioi_map, int_to_velocity_map,
                                                 m21.instrument.Flute()) # Melody as Flute
    melody_part.id = 'melody'
    
    harmony_part = symbolic_events_to_midi_stream(harmony_tuples,
                                                  int_to_pitch_map, int_to_duration_map,
                                                  int_to_ioi_map, int_to_velocity_map,
                                                  m21.instrument.Piano()) # Harmony as Piano
    harmony_part.id = 'harmony'
    
    score.insert(0, melody_part)
    score.insert(0, harmony_part)
    
    try:
        score.write('midi', fp=output_midi_path)
        print(f"Combined MIDI file saved to {output_midi_path}")
    except Exception as e:
        print(f"Error writing combined MIDI file: {e}")

print("Symbolic to MIDI conversion functions defined.")
# Make sure the int_to_... maps are loaded from Cell 5
if 'int_to_pitch' not in locals():
    print("Warning: int_to_pitch map not found. MIDI conversion might fail or use defaults.")
    # Create dummy maps if they don't exist to avoid NameErrors later, though generation won't be meaningful
    int_to_pitch = {0: '<PAD_PITCH>', **{i+1: i for i in range(128)}}
    int_to_duration = {0: '<PAD_DUR>', **{i+1: i*0.125 for i in range(1, 33)}}
    int_to_ioi = {0: '<PAD_IOI>', **{i+1: i*0.125 for i in range(1, 33)}}
    int_to_velocity = {0: '<PAD_VEL>', **{i+1: min(127, i*4) for i in range(1, 33)}}

# %%
# Cell 11: Conditioned Music Generation (Harmonization) Function & Sample Generation

def generate_harmony(model, condition_melody_raw_tuples, generation_length,
                     pitch_to_int_map, duration_to_int_map, ioi_to_int_map, velocity_to_int_map, # For condition
                     int_to_pitch_map, int_to_duration_map, int_to_ioi_map, int_to_velocity_map, # For output
                     device, temperature=1.0, max_condition_len=None):
    """
    Generates a harmony sequence conditioned on a raw melody sequence.
    """
    model.eval()

    # 1. Preprocess condition_melody_raw_tuples: tokenize
    # No padding needed here if encoder handles variable length.
    # If encoder expects fixed length, padding/truncation should be applied here.
    # For simplicity, let's assume the encoder handles variable length up to a certain max.
    # If not, we'd pad/truncate condition_melody_raw_tuples to MAX_SEQ_LENGTH (encoder's expected input length)
    
    tokenized_condition_melody = tokenize_event_sequence(
        condition_melody_raw_tuples, pitch_to_int_map, duration_to_int_map, ioi_to_int_map, velocity_to_int_map
    )
    
    # If encoder expects fixed length and condition is shorter than max_condition_len
    if max_condition_len and len(tokenized_condition_melody) < max_condition_len:
        tokenized_condition_melody += [PAD_EVENT_TOKEN_TUPLE] * (max_condition_len - len(tokenized_condition_melody))
    elif max_condition_len and len(tokenized_condition_melody) > max_condition_len:
         tokenized_condition_melody = tokenized_condition_melody[:max_condition_len]


    cond_p = torch.tensor([[event[0] for event in tokenized_condition_melody]], dtype=torch.long).to(device)
    cond_d = torch.tensor([[event[1] for event in tokenized_condition_melody]], dtype=torch.long).to(device)
    cond_i = torch.tensor([[event[2] for event in tokenized_condition_melody]], dtype=torch.long).to(device)
    cond_v = torch.tensor([[event[3] for event in tokenized_condition_melody]], dtype=torch.long).to(device)

    generated_harmony_token_tuples = [] # Store (p_tok, d_tok, i_tok, v_tok)

    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(cond_p, cond_d, cond_i, cond_v)

        # Initial decoder input: <SOS> tokens (e.g., PAD tokens or specific learned SOS)
        # For simplicity, using PAD tokens as SOS. First IOI often 0 for generated sequence.
        current_p_in = torch.full((1,), PAD_PITCH_TOKEN_ID, dtype=torch.long).to(device)
        current_d_in = torch.full((1,), PAD_DURATION_TOKEN_ID, dtype=torch.long).to(device)
        current_i_in = torch.full((1,), ioi_to_int_map.get(0.0, PAD_IOI_TOKEN_ID), dtype=torch.long).to(device) # Start with 0 IOI
        current_v_in = torch.full((1,), PAD_VELOCITY_TOKEN_ID, dtype=torch.long).to(device)


        for _ in range(generation_length):
            pred_p_logits, pred_d_logits, pred_i_logits, pred_v_logits, \
            hidden, cell, _ = model.decoder(
                current_p_in, current_d_in, current_i_in, current_v_in,
                hidden, cell, encoder_outputs
            )

            # Apply temperature and sample
            next_p_token = torch.multinomial(F.softmax(pred_p_logits / temperature, dim=-1), 1).squeeze().item()
            next_d_token = torch.multinomial(F.softmax(pred_d_logits / temperature, dim=-1), 1).squeeze().item()
            next_i_token = torch.multinomial(F.softmax(pred_i_logits / temperature, dim=-1), 1).squeeze().item()
            next_v_token = torch.multinomial(F.softmax(pred_v_logits / temperature, dim=-1), 1).squeeze().item()
            
            # Stop if PAD is generated for pitch (or other critical components)
            if next_p_token == PAD_PITCH_TOKEN_ID or next_d_token == PAD_DURATION_TOKEN_ID :
                 # Or if next_i_token means an excessive silence that indicates end of music
                print("Generation stopped due to PAD token or zero duration.")
                break
            
            generated_harmony_token_tuples.append((next_p_token, next_d_token, next_i_token, next_v_token))

            # Update inputs for the next step (autoregressive)
            current_p_in = torch.tensor([next_p_token], dtype=torch.long).to(device)
            current_d_in = torch.tensor([next_d_token], dtype=torch.long).to(device)
            current_i_in = torch.tensor([next_i_token], dtype=torch.long).to(device)
            current_v_in = torch.tensor([next_v_token], dtype=torch.long).to(device)
            
    # De-tokenize the generated harmony tokens
    detokenized_harmony = []
    for p_tok, d_tok, i_tok, v_tok in generated_harmony_token_tuples:
        p_val = int_to_pitch_map.get(p_tok)
        d_val = int_to_duration_map.get(d_tok)
        i_val = int_to_ioi_map.get(i_tok)
        v_val = int_to_velocity_map.get(v_tok)
        if any(val is None or (isinstance(val, str) and "<PAD" in val) for val in [p_val, d_val, i_val, v_val]) or d_val == 0:
            continue # Skip invalid or padding events
        detokenized_harmony.append((p_val, d_val, i_val, v_val))
        
    return detokenized_harmony


# --- Generate a sample harmony ---
# Load trained model if not already in memory
if 'cond_model' not in locals() or cond_model is None:
    try:
        print(f"Loading conditioned model from {MODEL_SAVE_PATH_COND}...")
        # Re-instantiate model architecture first
        attention_module_loaded = Attention(COND_ENC_HID_DIM, COND_DEC_HID_DIM)
        encoder_loaded = EncoderLSTM(pitch_vocab_size, duration_vocab_size, ioi_vocab_size, velocity_vocab_size,
                                     COND_EMB_PITCH, COND_EMB_DURATION, COND_EMB_IOI, COND_EMB_VELOCITY,
                                     COND_ENC_HID_DIM, COND_ENC_LAYERS, COND_DROPOUT)
        decoder_loaded = DecoderLSTM(pitch_vocab_size, duration_vocab_size, ioi_vocab_size, velocity_vocab_size,
                                     COND_EMB_PITCH, COND_EMB_DURATION, COND_EMB_IOI, COND_EMB_VELOCITY,
                                     COND_ENC_HID_DIM, COND_DEC_HID_DIM, COND_DEC_LAYERS, COND_DROPOUT, attention_module_loaded)
        cond_model = Seq2SeqMusic(encoder_loaded, decoder_loaded, device)
        cond_model.load_state_dict(torch.load(MODEL_SAVE_PATH_COND, map_location=device))
        cond_model.to(device)
        cond_model.eval()
        print("Conditioned model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Conditioned model checkpoint not found at {MODEL_SAVE_PATH_COND}. Cannot generate.")
        cond_model = None
    except Exception as e:
        print(f"Error loading conditioned model: {e}")
        cond_model = None


if cond_model and 'int_to_pitch' in locals() and raw_melody_harmony_data and raw_melody_harmony_data['test']:
    # Use a melody from the test set as condition
    # Take the first raw melody from the test set for simplicity
    sample_raw_melody_for_conditioning = raw_melody_harmony_data['test'][0]['melody']
    
    # Truncate/pad this raw melody to MAX_SEQ_LENGTH for consistency if desired, or let encoder handle
    # The generate_harmony function has a max_condition_len parameter now.
    
    print(f"\nUsing a sample melody from test set as condition (first 5 events): {sample_raw_melody_for_conditioning[:5]}")
    GENERATION_LENGTH_HARMONY = len(sample_raw_melody_for_conditioning) # Generate harmony of same length
    TEMPERATURE_HARMONY = 0.7

    print("Generating harmony for the sample melody...")
    generated_harmony_tuples = generate_harmony(
        cond_model,
        sample_raw_melody_for_conditioning, # Pass the raw (detokenized) melody
        GENERATION_LENGTH_HARMONY,
        pitch_to_int, duration_to_int, ioi_to_int, velocity_to_int, # value_to_int maps
        int_to_pitch, int_to_duration, int_to_ioi, int_to_velocity, # int_to_value maps
        device,
        temperature=TEMPERATURE_HARMONY,
        max_condition_len=MAX_SEQ_LENGTH # Encoder was trained with padded sequences
    )

    if generated_harmony_tuples:
        print(f"Generated {len(generated_harmony_tuples)} harmony events.")
        print(f"First 10 generated harmony events (values): {generated_harmony_tuples[:10]}")

        # Save the combined melody and generated harmony
        combined_midi_output_path = os.path.join(GENERATED_MIDI_DIR, "symbolic_conditioned_melody_harmony.mid")
        symbolic_to_midi_file_combined(
            sample_raw_melody_for_conditioning, # Original raw melody
            generated_harmony_tuples,          # Generated raw harmony
            combined_midi_output_path,
            int_to_pitch, int_to_duration, int_to_ioi, int_to_velocity
        )
    else:
        print("Harmony generation failed or produced an empty sequence.")
else:
    print("\nConditioned model, vocabularies, or test data not available. Skipping harmony generation.")

# %%
# Cell 12: Evaluation Metrics & Baseline for Harmonization

# Metric helper functions (calculate_basic_stats, calculate_polyphony_approx, get_pitch_class_distribution)
# are assumed to be defined from a previous notebook or earlier cell if following Piano_Generative.ipynb structure.
# For self-containment, let's redefine them here if they are not in the global scope.

# (If these functions were in a previous cell of *this* notebook, they'd be available)
# calculate_basic_stats, calculate_polyphony_approx, get_pitch_class_distribution ...

# --- Rule-Based Baseline Harmonizer ---
def generate_rule_based_harmony(melody_raw_tuples, interval=-12, default_velocity=64):
    """
    Simple rule-based harmonizer: plays a note a fixed interval below each melody note,
    using the melody's duration and IOI.
    """
    baseline_harmony_tuples = []
    for pitch, duration, ioi, _ in melody_raw_tuples: # Ignores melody velocity for baseline harmony
        if isinstance(pitch, (int, float)) and pitch >= 0: # Valid pitch
            harmony_pitch = pitch + interval
            if 0 <= harmony_pitch <= 127: # Ensure valid MIDI pitch
                 baseline_harmony_tuples.append((harmony_pitch, duration, ioi, default_velocity))
            else: # If interval makes it out of range, maybe play unison or omit
                 baseline_harmony_tuples.append((pitch, duration, ioi, default_velocity)) # Play unison as fallback
        # Rests or special tokens in melody could be mirrored or handled differently
    return baseline_harmony_tuples

# --- Calculate Metrics ---
metrics_results_conditioned = []

# 1. Metrics for Model-Generated Harmony
if 'generated_harmony_tuples' in locals() and generated_harmony_tuples:
    model_harmony_stats = calculate_basic_stats(generated_harmony_tuples)
    model_avg_poly, model_max_poly = calculate_polyphony_approx(generated_harmony_tuples)
    model_harmony_stats['avg_polyphony'] = model_avg_poly
    model_harmony_stats['max_polyphony'] = model_max_poly
    model_harmony_stats['source'] = 'Seq2Seq Model Harmony'
    metrics_results_conditioned.append(model_harmony_stats)
    print("\nMetrics for Seq2Seq Model Generated Harmony:")
    for k, v_ in model_harmony_stats.items(): print(f"  {k}: {v_:.2f}" if isinstance(v_, float) else f"  {k}: {v_}")
else:
    print("\n`generated_harmony_tuples` not found. Skipping its metrics.")

# 2. Metrics for Rule-Based Baseline Harmony
if 'sample_raw_melody_for_conditioning' in locals() and sample_raw_melody_for_conditioning:
    baseline_harmony_generated = generate_rule_based_harmony(sample_raw_melody_for_conditioning)
    if baseline_harmony_generated:
        baseline_stats = calculate_basic_stats(baseline_harmony_generated)
        baseline_avg_poly, baseline_max_poly = calculate_polyphony_approx(baseline_harmony_generated)
        baseline_stats['avg_polyphony'] = baseline_avg_poly
        baseline_stats['max_polyphony'] = baseline_max_poly
        baseline_stats['source'] = 'Rule-Based Baseline Harmony'
        metrics_results_conditioned.append(baseline_stats)
        print("\nMetrics for Rule-Based Baseline Harmony:")
        for k, v_ in baseline_stats.items(): print(f"  {k}: {v_:.2f}" if isinstance(v_, float) else f"  {k}: {v_}")
    else:
        print("\nRule-based baseline generation failed.")
else:
    print("\nSample melody for baseline not found. Skipping baseline metrics.")


# 3. Metrics for MAESTRO Test Set Harmonies (Ground Truth)
# We need to de-tokenize the processed_conditioned_sequences['test']'s 'target' parts
if processed_conditioned_sequences and processed_conditioned_sequences['test'] and 'int_to_pitch' in locals():
    test_set_harmonies_raw = []
    for item_dict in tqdm(processed_conditioned_sequences['test'], desc="De-tokenizing test set harmonies"):
        target_tensors = item_dict['target'] # tuple of (p,d,i,v) tensors
        
        # Convert tensors back to list of token tuples
        token_tuples_list = list(zip(
            target_tensors[0].tolist(), target_tensors[1].tolist(),
            target_tensors[2].tolist(), target_tensors[3].tolist()
        ))
        
        # De-tokenize
        raw_harmony_seq = []
        for p_tok, d_tok, i_tok, v_tok in token_tuples_list:
            if p_tok == PAD_PITCH_TOKEN_ID and d_tok == PAD_DURATION_TOKEN_ID: # Stop at first full PAD event
                break
            p_val = int_to_pitch.get(p_tok)
            d_val = int_to_duration.get(d_tok)
            i_val = int_to_ioi.get(i_tok)
            v_val = int_to_velocity.get(v_tok)
            if any(val is None or (isinstance(val, str) and "<PAD" in val) for val in [p_val, d_val, i_val, v_val]) or d_val == 0:
                continue
            raw_harmony_seq.append((p_val, d_val, i_val, v_val))
        if raw_harmony_seq: # Only add if not empty after de-tokenizing and removing padding
            test_set_harmonies_raw.append(raw_harmony_seq)

    if test_set_harmonies_raw:
        dataset_harmony_stats_list = []
        for seq in tqdm(test_set_harmonies_raw, desc="Calculating stats for test set harmonies"):
            stats = calculate_basic_stats(seq)
            avg_poly, max_poly = calculate_polyphony_approx(seq)
            stats['avg_polyphony'] = avg_poly
            stats['max_polyphony'] = max_poly
            dataset_harmony_stats_list.append(stats)
        
        if dataset_harmony_stats_list:
            df_dataset_harmony_stats = pd.DataFrame(dataset_harmony_stats_list)
            avg_dataset_harmony_stats = df_dataset_harmony_stats.mean().to_dict()
            avg_dataset_harmony_stats['source'] = 'MAESTRO Test Set Harmony (Avg)'
            metrics_results_conditioned.append(avg_dataset_harmony_stats)
            print("\nAverage Metrics for MAESTRO Test Set Harmonies:")
            for k, v_ in avg_dataset_harmony_stats.items(): print(f"  {k}: {v_:.2f}" if isinstance(v_, (float, np.floating)) else f"  {k}: {v_}")
    else:
        print("\nNo valid raw harmonies from test set to calculate stats.")
else:
    print("\nProcessed conditioned sequences (test split) or vocabularies not available. Skipping dataset harmony metrics.")

# Store all metrics in a DataFrame for easy comparison
df_metrics_summary_conditioned = pd.DataFrame(metrics_results_conditioned)
if not df_metrics_summary_conditioned.empty:
    df_metrics_summary_conditioned = df_metrics_summary_conditioned.set_index('source')
    print("\n--- Conditioned Task Metrics Summary ---")
    print(df_metrics_summary_conditioned)

# %%
# Cell 13: Comparison and Discussion for Harmonization Task

if 'df_metrics_summary_conditioned' in locals() and not df_metrics_summary_conditioned.empty:
    plot_df_cond = df_metrics_summary_conditioned.reset_index()

    # Select key metrics for plotting comparison
    metrics_to_plot_cond = ['num_notes', 'avg_pitch', 'avg_duration_ql', 'avg_ioi_ql', 'avg_velocity', 'avg_polyphony', 'unique_pitches']
    
    # Filter out metrics not present in the dataframe
    metrics_to_plot_cond = [m for m in metrics_to_plot_cond if m in plot_df_cond.columns]

    if metrics_to_plot_cond:
        plot_df_cond_melted = plot_df_cond.melt(id_vars='source', value_vars=metrics_to_plot_cond, 
                                                var_name='Metric', value_name='Value')

        plt.figure(figsize=(18, 10)) # Increased figure size
        sns.barplot(data=plot_df_cond_melted, x='Metric', y='Value', hue='source', palette='viridis')
        plt.title('Comparison of Musical Statistics for Generated Harmony')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.savefig(os.path.join(GENERATED_MIDI_DIR, "conditioned_metrics_comparison.png"))
        plt.show()
    else:
        print("No common metrics found to plot for conditioned task.")

    # Pitch Class Distribution Comparison
    # For Model Generated Harmony
    if 'generated_harmony_tuples' in locals() and generated_harmony_tuples:
        pc_dist_model_harmony = get_pitch_class_distribution(generated_harmony_tuples)
        df_pc_model_harmony = pd.DataFrame(list(pc_dist_model_harmony.items()), columns=['PitchClass', 'Count']).sort_values('PitchClass')
        
        plt.figure(figsize=(12, 5))
        sns.barplot(x='PitchClass', y='Count', data=df_pc_model_harmony, color=sns.color_palette("viridis")[0], order=range(12))
        plt.title('Pitch Class Distribution (Seq2Seq Model Harmony)')
        plt.xlabel('Pitch Class (0=C, ..., 11=B)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(GENERATED_MIDI_DIR, "pc_dist_model_harmony.png"))
        plt.show()

    # For Rule-Based Baseline Harmony
    if 'baseline_harmony_generated' in locals() and baseline_harmony_generated:
        pc_dist_baseline_harmony = get_pitch_class_distribution(baseline_harmony_generated)
        df_pc_baseline_harmony = pd.DataFrame(list(pc_dist_baseline_harmony.items()), columns=['PitchClass', 'Count']).sort_values('PitchClass')

        plt.figure(figsize=(12, 5))
        sns.barplot(x='PitchClass', y='Count', data=df_pc_baseline_harmony, color=sns.color_palette("viridis")[1], order=range(12))
        plt.title('Pitch Class Distribution (Rule-Based Baseline Harmony)')
        plt.xlabel('Pitch Class')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(GENERATED_MIDI_DIR, "pc_dist_baseline_harmony.png"))
        plt.show()

    # For MAESTRO Test Set Harmonies (Average or a sample)
    if 'test_set_harmonies_raw' in locals() and test_set_harmonies_raw:
        # Aggregate pitch classes from all test set harmonies
        all_test_harmony_pitches = []
        for seq in test_set_harmonies_raw:
            for event in seq:
                 if isinstance(event[0], (int, float)) and event[0] >=0:
                    all_test_harmony_pitches.append(event[0] % 12)
        
        if all_test_harmony_pitches:
            pc_dist_test_harmonies = Counter(all_test_harmony_pitches)
            # Normalize by number of sequences or total notes for a fair comparison if desired
            # For now, raw counts:
            df_pc_test_harmonies = pd.DataFrame(list(pc_dist_test_harmonies.items()), columns=['PitchClass', 'Count']).sort_values('PitchClass')

            plt.figure(figsize=(12, 5))
            sns.barplot(x='PitchClass', y='Count', data=df_pc_test_harmonies, color=sns.color_palette("viridis")[2], order=range(12))
            plt.title('Aggregated Pitch Class Distribution (MAESTRO Test Set Harmonies)')
            plt.xlabel('Pitch Class')
            plt.ylabel('Total Frequency')
            plt.savefig(os.path.join(GENERATED_MIDI_DIR, "pc_dist_test_harmonies.png"))
            plt.show()
        else:
            print("No pitch data to plot for test set harmonies.")
else:
    print("Metrics summary for conditioned task is empty or not defined. Skipping comparison plotting.")


