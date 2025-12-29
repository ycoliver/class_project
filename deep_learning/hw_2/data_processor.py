import os
import json
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm

def process_librispeech(data_dir, output_file, target_sr=16000, subset_limit=None):
    """
    Process LibriSpeech dataset: pair audio files with transcriptions and save to jsonl.
    
    Args:
        data_dir: Path to LibriSpeech data directory (e.g., 'LibriSpeech/test-clean')
        output_file: Output jsonl file path
        target_sr: Target sampling rate (default: 16000 Hz)
        subset_limit: Optional limit on number of samples to process (for testing)
    """
    data_pairs = []
    processed_count = 0
    
    print(f"Processing LibriSpeech data from: {data_dir}")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(data_dir):
        # Find transcript files (*.trans.txt)
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        
        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            
            # Read all transcriptions from the file
            transcriptions = {}
            with open(trans_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Split at first space: ID + transcription
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            audio_id, text = parts
                            transcriptions[audio_id] = text
            
            # Process each audio file
            for audio_id, transcription in tqdm(transcriptions.items()):
                # Find corresponding audio file (.flac)
                audio_file = f"{audio_id}.flac"
                audio_path = os.path.join(root, audio_file)
                
                if os.path.exists(audio_path):
                    try:
                        # Load audio
                        waveform, sample_rate = torchaudio.load(audio_path)
                        
                        # Resample to target sample rate if needed
                        if sample_rate != target_sr:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=sample_rate,
                                new_freq=target_sr
                            )
                            waveform = resampler(waveform)
                        
                        # Convert to mono if stereo
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                        # Create data pair
                        data_pair = {
                            'audio_id': audio_id,
                            'audio_path': audio_path,
                            'text': transcription,
                            'sample_rate': target_sr,
                            'duration': waveform.shape[1] / target_sr,
                            'num_samples': waveform.shape[1]
                        }
                        
                        data_pairs.append(data_pair)
                        processed_count += 1
                        
                        # Check if we've reached the subset limit
                        if subset_limit and processed_count >= subset_limit:
                            print(f"Reached subset limit of {subset_limit} samples")
                            break
                            
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
                else:
                    print(f"Audio file not found: {audio_path}")
            
            if subset_limit and processed_count >= subset_limit:
                break
        
        if subset_limit and processed_count >= subset_limit:
            break
    
    # Save to jsonl file
    print(f"\nSaving {len(data_pairs)} samples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in data_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Processing complete! Total samples: {len(data_pairs)}")
    return data_pairs


def visualize_sample(data_dir, output_dir='visualizations', num_samples=3):
    """
    Visualize Mel-spectrograms for a few random samples.
    
    Args:
        data_dir: Path to LibriSpeech data directory
        output_dir: Directory to save visualization plots
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect some samples
    samples = []
    for root, dirs, files in os.walk(data_dir):
        audio_files = [f for f in files if f.endswith('.flac')]
        for audio_file in audio_files[:num_samples]:
            audio_path = os.path.join(root, audio_file)
            samples.append(audio_path)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # Create Mel-spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )
    
    for idx, audio_path in enumerate(samples):
        try:
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Compute Mel-spectrogram
            mel_spec = mel_transform(waveform)
            mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Waveform
            axes[0].plot(waveform[0].numpy())
            axes[0].set_title(f'Waveform - {os.path.basename(audio_path)}')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('Amplitude')
            
            # Mel-spectrogram
            im = axes[1].imshow(mel_spec_db[0].numpy(), aspect='auto', origin='lower', 
                               cmap='viridis')
            axes[1].set_title('Mel-Spectrogram')
            axes[1].set_xlabel('Time Frame')
            axes[1].set_ylabel('Mel Frequency Bin')
            plt.colorbar(im, ax=axes[1], format='%+2.0f dB')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'sample_{idx+1}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization: {output_path}")
            
        except Exception as e:
            print(f"Error visualizing {audio_path}: {e}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "./datasets/LibriSpeech"
    
    # Process test-clean dataset
    test_clean_dir = os.path.join(BASE_DIR, "test-clean")
    test_output = "test_clean.jsonl"
    
    print("="*60)
    print("Processing test-clean dataset")
    print("="*60)
    
    # Process full test-clean or use subset_limit for testing
    # For testing, you can set subset_limit=100 to process only 100 samples
    print('Processing test-clean samples:')
    test_data = process_librispeech(
        data_dir=test_clean_dir,
        output_file=test_output,
        target_sr=16000,
        subset_limit=None  # Set to a number (e.g., 100) to limit samples
    )
    
    # Process train-clean-100 dataset (optional, can use subset)
    train_clean_dir = os.path.join(BASE_DIR, "train-clean-100")
    train_output = "train_clean_100.jsonl"
    
    if os.path.exists(train_clean_dir):
        print("\n" + "="*60)
        print("Processing train-clean-100 dataset")
        print("="*60)
        print('Processing train-clean samples:')
        # For training, you might want to use a subset for faster experimentation
        train_data = process_librispeech(
            data_dir=train_clean_dir,
            output_file=train_output,
            target_sr=16000,
            subset_limit=None  # Use 1000 samples for quick training
        )
    
    # Visualize some samples
    print("\n" + "="*60)
    print("Creating visualizations")
    print("="*60)
    visualize_sample(test_clean_dir, output_dir='visualizations', num_samples=3)
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)