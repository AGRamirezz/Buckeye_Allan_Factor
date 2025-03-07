# -*- coding: utf-8 -*-
"""
Data Processing for Buckeye Corpus Analysis

This script processes phonetic and word data from the Buckeye Corpus to analyze
speech timing patterns across different linguistic units (phones, syllables, words, etc.).
The analysis focuses on calculating speech rates, variability metrics, and duration patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from statistics import mean, median, stdev
from scipy import stats
import scipy.io as sio
import h5py

# Define constants
DATA_ROOT = '/Users/cogmech/Documents/Buckeye_AF/Data'
PHONES_DIR = f"{DATA_ROOT}/text/phones"
WORDS_DIR = f"{DATA_ROOT}/text/words"
AUDIO_DIR = f"{DATA_ROOT}/audio"
CHUNK_DURATION = 240  # 4 minutes in seconds


def load_phone_data():
    """
    Load and preprocess phone data from text files.
    
    Returns:
        List of dataframes containing processed phone data
    """
    phones = []
    
    os.chdir(PHONES_DIR)
    for filename in os.listdir(os.getcwd()):
        if not filename.endswith(".txt"):
            continue
            
        # Read file with appropriate column separation
        current_frame = pd.read_csv(
            filename, 
            header=None, 
            skiprows=9,
            sep=r'\s+',
            error_bad_lines=False,
            usecols=[0, 2]
        )
        
        # Clean up the data - remove brackets and convert to string
        current_frame[2] = current_frame[2].astype(str)
        current_frame[2] = current_frame[2].str.lstrip('{').str.rstrip('};')
        
        # Add a column to identify phones vs. non-speech labels (nsl)
        # Use vectorized operations instead of loop
        current_frame[1] = 'nsl'
        current_frame.loc[current_frame[2].str.islower(), 1] = 'phone'
        
        phones.append(current_frame)
    
    return phones


def load_word_data():
    """
    Load and preprocess word data from text files.
    
    Returns:
        List of dataframes containing processed word data
    """
    words = []
    
    os.chdir(WORDS_DIR)
    for filename in os.listdir(os.getcwd()):
        if not filename.endswith(".txt"):
            continue
            
        # Read file with appropriate column separation
        current_frame = pd.read_csv(
            filename, 
            header=None, 
            skiprows=9,
            sep=r'\s+',
            error_bad_lines=False,
            usecols=[0, 2]
        )
        
        # Clean up the data - remove brackets and convert to string
        current_frame[2] = current_frame[2].str.lstrip('<{').str.rstrip('>};')
        
        # Add a column to identify words vs. non-speech labels (nsl)
        # Use vectorized operations instead of loop
        current_frame[1] = 'nsl'
        current_frame.loc[current_frame[2].str.islower(), 1] = 'word'
        
        words.append(current_frame)
    
    return words


def get_participant_ids():
    """
    Extract participant IDs from filenames.
    
    Returns:
        List of participant IDs (first 3 chars of filenames)
    """
    return [item[0:3] for item in os.listdir(os.getcwd()) if item.endswith(".txt")]


def merge_participant_files(data_list, participant_ids):
    """
    Merge files belonging to the same participant into a single dataframe.
    
    Args:
        data_list: List of dataframes to merge
        participant_ids: List of participant IDs from filenames
    
    Returns:
        List of dataframes, one per participant
    """
    # Find indices where participant changes
    change_indices = [i for i in range(len(participant_ids)-1) 
                     if participant_ids[i] != participant_ids[i+1]]
    
    # Add start and end indices
    all_indices = [0] + change_indices + [len(data_list)]
    
    merged_parts = []
    
    # Process each participant's data
    for i in range(len(all_indices)-1):
        start_idx = all_indices[i]
        end_idx = all_indices[i+1]
        
        # Get files for current participant
        if i == 0:
            part = data_list[start_idx:end_idx+1]
        else:
            part = data_list[start_idx+1:end_idx+1]
        
        # Adjust timestamps to make them continuous across files
        for j in range(len(part)-1):
            precol_max = part[j][0].max()
            part[j+1][0] += precol_max
        
        # Concatenate all files for this participant
        merged_parts.append(pd.concat(part, ignore_index=True))
    
    return merged_parts


def split_into_chunks(part_data, first_row=None, last_row=None):
    """
    Split participant data into fixed-duration chunks (e.g., 4 minutes = 240 seconds).
    
    Args:
        part_data: List of participant dataframes
        first_row: Row to add at the beginning of each chunk
        last_row: Row to add at the end of each chunk
    
    Returns:
        List of lists of dataframes, chunked by participant and time
    """
    part_chunks = []
    
    for participant_df in part_data:
        chunks = []
        
        # Calculate number of chunks based on max timestamp
        max_stamp = participant_df[0].max()
        num_chunks = math.floor(max_stamp / CHUNK_DURATION)
        
        if num_chunks == 0:
            continue
            
        df_length = len(participant_df) - 1
        chunk_length = math.floor(df_length / num_chunks)
        
        # Create chunks
        for j in range(chunk_length, df_length, chunk_length):
            # Extract chunk
            chunk_df = participant_df.iloc[j-chunk_length:j].reset_index(drop=True)
            
            # Add delimiter rows at start and end if provided
            if first_row is not None:
                chunk_df = pd.concat([pd.DataFrame([first_row]), chunk_df], ignore_index=True)
            if last_row is not None:
                chunk_df = pd.concat([chunk_df, pd.DataFrame([last_row])], ignore_index=True)
            
            chunks.append(chunk_df)
        
        part_chunks.append(chunks)
    
    return part_chunks


def analyze_durations(chunks, unit_type='phone'):
    """
    Calculate durations between consecutive units (phones, words, etc.).
    
    Args:
        chunks: List of lists of dataframes
        unit_type: Type of unit to analyze ('phone', 'word', etc.)
    
    Returns:
        List of lists of duration measurements
    """
    all_diffs = []
    
    for participant_chunks in chunks:
        participant_diffs = []
        
        for chunk_df in participant_chunks:
            durations = []
            
            # Find consecutive units of the specified type
            mask = chunk_df[1] == unit_type
            unit_indices = chunk_df.index[mask].tolist()
            
            # Calculate durations between consecutive units
            for i in range(len(unit_indices)-1):
                if unit_indices[i+1] - unit_indices[i] == 1:  # Consecutive units
                    onset = chunk_df.loc[unit_indices[i], 0]
                    offset = chunk_df.loc[unit_indices[i+1], 0]
                    durations.append(offset - onset)
            
            participant_diffs.append(durations)
        
        all_diffs.append(participant_diffs)
    
    return all_diffs


def calculate_speech_rate(duration_data):
    """
    Calculate speech rate (units per second) for each chunk.
    
    Args:
        duration_data: List of lists of duration measurements
    
    Returns:
        List of speech rates
    """
    rates = []
    
    for participant_data in duration_data:
        for chunk_durations in participant_data:
            # Rate = number of units / chunk duration
            rate = len(chunk_durations) / CHUNK_DURATION
            rates.append(rate)
    
    return rates


def calculate_variability_metrics(duration_data):
    """
    Calculate coefficient of variation and standard deviation for durations.
    
    Args:
        duration_data: List of lists of duration measurements
    
    Returns:
        Tuple of (coefficient of variation, standard deviation)
    """
    cof_values = []
    std_values = []
    
    for participant_data in duration_data:
        participant_cof = []
        participant_std = []
        
        for chunk_durations in participant_data:
            if not chunk_durations:  # Skip empty chunks
                continue
                
            # Calculate coefficient of variation (CV = std/mean)
            cof = stats.variation(chunk_durations)
            participant_cof.append(cof)
            
            # Calculate standard deviation
            std = np.std(chunk_durations)
            participant_std.append(std)
        
        cof_values.append(np.array(participant_cof))
        std_values.append(np.array(participant_std))
    
    return np.array(cof_values), np.array(std_values)


def label_breath_groups(word_chunks):
    """
    Label breath groups in word data based on consecutive words.
    
    Args:
        word_chunks: List of lists of word dataframes
    
    Returns:
        Updated word_chunks with breath group labels added
    """
    for participant_chunks in word_chunks:
        for chunk_df in participant_chunks:
            # Add breath group column
            chunk_df['bgs'] = 'out'  # Default to out-of-breath-group
            
            # Find consecutive words and mark them as in-breath-group
            for j in range(len(chunk_df)-1):
                if chunk_df.loc[j, 1] == 'word' and chunk_df.loc[j+1, 1] == 'word':
                    chunk_df.loc[j, 'bgs'] = 'in'
    
    return word_chunks


def analyze_breath_group_durations(word_chunks):
    """
    Calculate breath group durations from labeled word data.
    
    Args:
        word_chunks: List of lists of word dataframes with breath group labels
    
    Returns:
        List of lists of breath group duration measurements
    """
    all_bg_diffs = []
    
    for participant_chunks in word_chunks:
        bg_diffs = []
        
        for chunk_df in participant_chunks:
            durations = []
            onset = None
            
            for j in range(1, len(chunk_df)):
                # Identify breath group onsets (transition from out to in)
                if chunk_df.loc[j, 'bgs'] == 'in' and chunk_df.loc[j-1, 'bgs'] == 'out':
                    onset = chunk_df.loc[j, 0]
                
                # Identify breath group offsets (last word before pause)
                elif j+1 < len(chunk_df) and chunk_df.loc[j, 'bgs'] == 'in' and chunk_df.loc[j+1, 'bgs'] == 'out':
                    if onset is not None:  # Make sure we have an onset
                        offset = chunk_df.loc[j, 0]
                        durations.append(offset - onset)
                        onset = None
            
            bg_diffs.append(durations)
        
        all_bg_diffs.append(bg_diffs)
    
    return all_bg_diffs


def analyze_pos_durations(pos_chunks, pos_list):
    """
    Calculate durations of words with specific parts of speech.
    
    Args:
        pos_chunks: List of lists of dataframes with POS tags
        pos_list: List of POS tags to analyze
    
    Returns:
        List of duration measurements for words with specified POS tags
    """
    pos_diffs = []
    
    for participant_chunks in pos_chunks:
        for chunk_df in participant_chunks:
            chunk_durations = []
            
            for k in range(len(chunk_df)-1):
                # Check if current word has the specified POS and next word exists
                if chunk_df.loc[k, 3] in pos_list and chunk_df.loc[k+1, 3] != 'null':
                    onset = chunk_df.loc[k, 0]
                    offset = chunk_df.loc[k+1, 0]
                    duration = offset - onset
                    chunk_durations.append(duration)
            
            pos_diffs.append(chunk_durations)
    
    return pos_diffs


def export_data_to_csv(data_dict, filename):
    """
    Export data to CSV file.
    
    Args:
        data_dict: Dictionary of data to export
        filename: Name of the CSV file
    """
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(filename)
    print(f"Data exported to {filename}")


def main():
    """Main function to run all analyses."""
    print("Starting Buckeye Corpus analysis...")
    
    # Process phone data
    print("Processing phone data...")
    phones = load_phone_data()
    os.chdir(PHONES_DIR)
    participant_ids = get_participant_ids()
    
    phone_parts = merge_participant_files(phones, participant_ids)
    
    # Create delimiter rows for chunking
    dt_row = {0: 0, 1: 'nsl', 2: 'NaN'}
    db_row = phones[0].iloc[-1].to_dict()
    
    phone_chunks = split_into_chunks(phone_parts, first_row=dt_row, last_row=db_row)
    phone_diffs = analyze_durations(phone_chunks, unit_type='phone')
    phone_rate = calculate_speech_rate(phone_diffs)
    phone_cof, phone_std = calculate_variability_metrics(phone_diffs)
    
    # Process word data
    print("Processing word data...")
    words = load_word_data()
    os.chdir(WORDS_DIR)
    participant_ids = get_participant_ids()
    
    word_parts = merge_participant_files(words, participant_ids)
    
    # Create delimiter rows for chunking
    dt_row = words[0].iloc[0].to_dict()
    db_row = words[0].iloc[-1].to_dict()
    
    word_chunks = split_into_chunks(word_parts, first_row=dt_row, last_row=db_row)
    word_diffs = analyze_durations(word_chunks, unit_type='word')
    word_rate = calculate_speech_rate(word_diffs)
    word_cof, word_std = calculate_variability_metrics(word_diffs)
    
    # Process breath groups
    print("Analyzing breath groups...")
    word_chunks = label_breath_groups(word_chunks)
    bg_diffs = analyze_breath_group_durations(word_chunks)
    bg_rate = calculate_speech_rate(bg_diffs)
    bg_cof, bg_std = calculate_variability_metrics(bg_diffs)
    
    # Load acoustic features from MAT file
    print("Loading acoustic features...")
    os.chdir(AUDIO_DIR)
    with h5py.File('audio_af.mat', 'r') as f:
        lin_slope = f['lin'][0][:]
        quad_coef = f['quad'][0][:]
        quad_slope = f['quad'][1][:]
        acoustic_layers = f['lays'][:][:]
    
    # Export results
    print("Exporting results...")
    
    # Export rate data
    rate_data = {
        'phone_rate': phone_rate,
        'word_rate': word_rate,
        'bg_rate': bg_rate
    }
    export_data_to_csv(rate_data, 'rates_data.csv')
    
    # Export variability data
    cof_data = {
        'phone_cof': np.concatenate(phone_cof),
        'word_cof': np.concatenate(word_cof),
        'bg_cof': np.concatenate(bg_cof),
        'lin_slope': lin_slope,
        'quad_coef': quad_coef,
        'quad_slope': quad_slope
    }
    
    # Add acoustic layers to data dictionary
    for i in range(acoustic_layers.shape[0]):
        cof_data[f'ay{i+1}'] = acoustic_layers[i][:]
    
    export_data_to_csv(cof_data, 'all_data.csv')
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

