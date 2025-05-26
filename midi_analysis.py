import os
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIDIAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize the MIDI analyzer with the data directory."""
        self.data_dir = Path(data_dir)
        self.midi_files = list(self.data_dir.rglob("*.midi"))
        logger.info(f"Found {len(self.midi_files)} MIDI files")
        
        # Initialize storage for features
        self.features = []
        self.stats = {}
        
    def extract_basic_features(self, midi_path: Path) -> Dict:
        """Extract basic features from a single MIDI file."""
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Get piano roll
            piano_roll = pm.get_piano_roll(fs=4)  # 4 Hz resolution
            
            # Basic features
            features = {
                'file_name': midi_path.name,
                'duration': pm.get_end_time(),
                'num_instruments': len(pm.instruments),
                'num_notes': sum(len(inst.notes) for inst in pm.instruments),
                'mean_velocity': np.mean([note.velocity for inst in pm.instruments for note in inst.notes]),
                'std_velocity': np.std([note.velocity for inst in pm.instruments for note in inst.notes]),
                'mean_duration': np.mean([note.end - note.start for inst in pm.instruments for note in inst.notes]),
                'std_duration': np.std([note.end - note.start for inst in pm.instruments for note in inst.notes]),
                'mean_pitch': np.mean([note.pitch for inst in pm.instruments for note in inst.notes]),
                'std_pitch': np.std([note.pitch for inst in pm.instruments for note in inst.notes]),
                'polyphony': np.mean(np.sum(piano_roll > 0, axis=0)),  # Average number of notes played simultaneously
                'note_density': len(pm.instruments[0].notes) / pm.get_end_time() if pm.instruments else 0,
            }
            
            # Add tempo information
            
            if pm.get_tempo_changes():
                # Get tempo changes as (time, tempo) pairs
                tempo_changes = pm.get_tempo_changes()
                times, tempos = tempo_changes[0], tempo_changes[1]
                
                # Calculate weighted average tempo based on duration between changes
                durations = np.diff(times, append=pm.get_end_time())
                features['mean_tempo'] = np.average(tempos, weights=durations)
                
                # Calculate weighted standard deviation
                weighted_mean = features['mean_tempo']
                weighted_variance = np.average((tempos - weighted_mean)**2, weights=durations)
                features['std_tempo'] = np.sqrt(weighted_variance)
            else:
                features['mean_tempo'] = 120  # Default tempo
                features['std_tempo'] = 0
                
            return features
            
        except Exception as e:
            logger.error(f"Error processing {midi_path}: {str(e)}")
            return None

    def process_all_files(self, n_workers: int = 4):
        """Process all MIDI files in parallel."""
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_basic_features, self.midi_files),
                total=len(self.midi_files),
                desc="Processing MIDI files"
            ))
        
        # Filter out None results and store features
        self.features = [f for f in results if f is not None]
        logger.info(f"Successfully processed {len(self.features)} files")
        
    def compute_statistics(self):
        """Compute overall statistics from the extracted features."""
        if not self.features:
            logger.warning("No features available. Run process_all_files first.")
            return
            
        df = pd.DataFrame(self.features)
        
        # Compute basic statistics
        self.stats = {
            'total_files': len(df),
            'mean_duration': df['duration'].mean(),
            'mean_notes_per_file': df['num_notes'].mean(),
            'mean_polyphony': df['polyphony'].mean(),
            'mean_note_density': df['note_density'].mean(),
            'mean_tempo': df['mean_tempo'].mean(),
            'pitch_range': {
                'min': df['mean_pitch'].min() - 2 * df['std_pitch'].min(),
                'max': df['mean_pitch'].max() + 2 * df['std_pitch'].max()
            }
        }
        
        logger.info("Statistics computed successfully")
        
    def plot_distributions(self, save_dir: str = 'plots'):
        """Create and save distribution plots of key features."""
        if not self.features:
            logger.warning("No features available. Run process_all_files first.")
            return
            
        df = pd.DataFrame(self.features)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create plots for key features
        features_to_plot = [
            'duration', 'num_notes', 'mean_velocity', 'mean_duration',
            'mean_pitch', 'polyphony', 'note_density', 'mean_tempo'
        ]
        
        for feature in features_to_plot:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, bins=50)
            plt.title(f'Distribution of {feature}')
            plt.savefig(save_path / f'{feature}_distribution.png')
            plt.close()
            
        logger.info(f"Plots saved to {save_path}")
        
    def save_features(self, output_file: str = 'midi_features.json'):
        """Save the extracted features and statistics to a JSON file."""
        output = {
            'statistics': self.stats,
            'features': self.features
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Features saved to {output_file}")

def main():
    # Initialize analyzer
    analyzer = MIDIAnalyzer('data/maestro-v3.0.0')
    
    # Process all files
    analyzer.process_all_files()
    
    # Compute statistics
    analyzer.compute_statistics()
    
    # Create visualizations
    analyzer.plot_distributions()
    
    # Save features
    analyzer.save_features()
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Total files processed: {analyzer.stats['total_files']}")
    print(f"Average duration: {analyzer.stats['mean_duration']:.2f} seconds")
    print(f"Average notes per file: {analyzer.stats['mean_notes_per_file']:.2f}")
    print(f"Average polyphony: {analyzer.stats['mean_polyphony']:.2f}")
    print(f"Average note density: {analyzer.stats['mean_note_density']:.2f} notes/second")
    print(f"Average tempo: {analyzer.stats['mean_tempo']:.2f} BPM")

if __name__ == "__main__":
    main() 