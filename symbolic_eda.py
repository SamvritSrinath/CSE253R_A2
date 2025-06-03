import os
import numpy as np
import pandas as pd
import pretty_midi
import music21
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import json
from collections import Counter
from scipy.stats import entropy
import warnings
from miditok import MIDILike
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SymbolicAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize the symbolic analyzer with the data directory."""
        self.data_dir = Path(data_dir)
        self.midi_files = list(self.data_dir.rglob("*.midi"))
        if not self.midi_files:
            self.midi_files = list(self.data_dir.rglob("*.mid"))
        logger.info(f"Found {len(self.midi_files)} MIDI files")
        
        # Initialize miditok tokenizer
        self.tokenizer = MIDILike(
            pitch_range=(21, 109),  # Piano range
            beat_res={(0, 4): 8, (4, 12): 4},  # 8 positions per quarter note, 4 per eighth note
            nb_velocities=32,  # 32 velocity bins
            additional_tokens={
                "Chord": True,  # Include chord tokens
                "Rest": True,   # Include rest tokens
                "Tempo": True,  # Include tempo tokens
                "TimeSignature": True,  # Include time signature tokens
            }
        )
        
        self.features = []
        self.stats = {}
        
    def extract_musical_features(self, midi_path: Path) -> Dict:
        """Extract comprehensive musical features from a single MIDI file using music21 and miditok."""
        try:
            # Load with music21 for musical analysis
            score = music21.converter.parse(str(midi_path))
            
            # Basic features
            features = {
                'file_name': midi_path.name,
                'duration': score.duration.quarterLength,
                'num_measures': len(score.getElementsByClass('Measure')),
                'num_notes': len(score.flat.notes),
            }
            
            # Get all notes
            notes = list(score.flat.notes)
            if notes:
                # Basic note statistics
                pitches = [note.pitch.midi for note in notes]
                velocities = [note.volume.velocity if note.volume.velocity else 64 for note in notes]
                durations = [note.quarterLength for note in notes]
                
                features.update({
                    'mean_velocity': np.mean(velocities),
                    'std_velocity': np.std(velocities),
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'mean_pitch': np.mean(pitches),
                    'std_pitch': np.std(pitches),
                })
                
                # Advanced pitch features
                pitch_classes = [p % 12 for p in pitches]
                pc_counts = np.bincount(pitch_classes, minlength=12)
                if np.sum(pc_counts) > 0:
                    features['pitch_class_entropy'] = entropy(pc_counts / np.sum(pc_counts))
                features['pitch_range'] = max(pitches) - min(pitches)
                features['unique_pitches'] = len(set(pitches))
                features['unique_pitch_classes'] = len(set(pitch_classes))
                
                # Rhythmic features
                if len(notes) > 1:
                    note_starts = sorted([note.offset for note in notes])
                    iois = np.diff(note_starts)
                    iois = iois[iois > 0]
                    if iois.size > 0:
                        ioi_hist_counts, _ = np.histogram(iois, bins=50)
                        if np.sum(ioi_hist_counts) > 0:
                            features['ioi_entropy'] = entropy(ioi_hist_counts / np.sum(ioi_hist_counts))
                        features.update({
                            'mean_ioi': np.mean(iois),
                            'std_ioi': np.std(iois),
                        })
                
                # Melodic features
                if len(notes) > 1:
                    intervals = []
                    for i in range(len(notes) - 1):
                        interval = notes[i + 1].pitch.midi - notes[i].pitch.midi
                        intervals.append(interval)
                    
                    if intervals:
                        intervals = np.array(intervals)
                        hist_counts, _ = np.histogram(intervals, bins=50)
                        if np.sum(hist_counts) > 0:
                            features['interval_entropy'] = entropy(hist_counts / np.sum(hist_counts))
                        features.update({
                            'mean_interval': np.mean(intervals),
                            'std_interval': np.std(intervals),
                            'melodic_range': np.ptp(intervals),
                        })
            
            # Key analysis
            key_analysis = score.analyze('key')
            features.update({
                'key_mode': key_analysis.mode,
                'key_tonic': key_analysis.tonic.name,
                'key_correlation': key_analysis.correlationCoefficient,
            })
            
            # Chord analysis
            chord_analysis = score.chordify()
            chord_types = []
            for chord in chord_analysis.recurse().getElementsByClass('Chord'):
                chord_types.append(chord.commonName)
            
            if chord_types:
                chord_counter = Counter(chord_types)
                most_common_chord = chord_counter.most_common(1)[0]
                features.update({
                    'most_common_chord': most_common_chord[0],
                    'chord_diversity': len(set(chord_types)) / len(chord_types),
                    'chord_entropy': entropy(np.array(list(chord_counter.values())) / len(chord_types)),
                })
            
            # Time signature analysis
            time_signatures = score.getTimeSignatures()
            if time_signatures:
                features['time_signature'] = f"{time_signatures[0].numerator}/{time_signatures[0].denominator}"
            
            # Tempo analysis
            tempos = score.metronomeMarkBoundaries()
            if tempos:
                tempo_values = [t[2].number for t in tempos]
                tempo_durations = [t[1] - t[0] for t in tempos]
                if tempo_durations:
                    features.update({
                        'mean_tempo': np.average(tempo_values, weights=tempo_durations),
                        'std_tempo': np.sqrt(np.average((np.array(tempo_values) - np.average(tempo_values, weights=tempo_durations))**2, weights=tempo_durations)),
                        'tempo_changes': len(tempos),
                    })
            
            # Tokenization analysis using miditok
            try:
                tokens = self.tokenizer.midi_to_tokens(score)
                if tokens:
                    features.update({
                        'num_tokens': len(tokens[0]),  # Assuming single track
                        'token_types': len(set(tokens[0])),
                        'token_entropy': entropy(np.bincount(tokens[0], minlength=self.tokenizer.vocab_size) / len(tokens[0])),
                    })
            except Exception as e:
                logger.warning(f"Tokenization failed for {midi_path.name}: {str(e)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {midi_path.name} ({type(e).__name__}): {str(e)}", exc_info=False)
            return None

    def process_all_files(self, n_workers: Optional[int] = None):
        """Process all MIDI files in parallel."""
        if n_workers is None:
            n_workers = os.cpu_count()
            logger.info(f"Using {n_workers} workers (all available CPU cores).")
            
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_musical_features, self.midi_files),
                total=len(self.midi_files),
                desc="Processing MIDI files"
            ))
        
        self.features = [f for f in results if f is not None]
        logger.info(f"Successfully processed {len(self.features)} files")
        
    def compute_statistics(self):
        """Compute comprehensive statistics from the extracted features."""
        if not self.features:
            logger.warning("No features available. Run process_all_files first.")
            return
            
        df = pd.DataFrame(self.features)
        
        self.stats = {
            'total_files_processed': len(df),
            'total_duration_hours': df['duration'].sum() / 3600,
            'mean_duration': df['duration'].mean(),
            'mean_notes_per_file': df['num_notes'].mean(),
            'key_distribution': df['key_tonic'].value_counts().to_dict(),
            'mode_distribution': df['key_mode'].value_counts().to_dict(),
            'time_signature_distribution': df['time_signature'].value_counts().to_dict(),
            'chord_diversity': df['chord_diversity'].mean(),
            'melodic_complexity': {
                'mean_interval': df['mean_interval'].mean(),
                'interval_entropy': df['interval_entropy'].mean(),
            },
            'rhythmic_complexity': {
                'mean_ioi': df['mean_ioi'].mean(),
                'ioi_entropy': df['ioi_entropy'].mean(),
            },
            'tokenization_stats': {
                'mean_tokens_per_file': df['num_tokens'].mean(),
                'mean_token_types': df['token_types'].mean(),
                'mean_token_entropy': df['token_entropy'].mean(),
            }
        }
        
        logger.info("Statistics computed successfully")
        
    def plot_distributions(self, save_dir: str = 'plots'):
        """Create and save comprehensive distribution plots of features."""
        if not self.features:
            logger.warning("No features available. Run process_all_files first.")
            return
            
        df = pd.DataFrame(self.features)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Basic Features
        basic_features = [
            'duration', 'num_notes', 'mean_velocity', 'mean_duration',
            'mean_pitch', 'mean_tempo'
        ]
        
        for feature in basic_features:
            if feature in df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=feature, kde=True, bins=50)
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
                plt.savefig(save_path / f'{feature}_distribution.png')
                plt.close()
        
        # 2. Musical Features
        # Key Distribution
        if 'key_tonic' in df.columns:
            plt.figure(figsize=(12, 6))
            key_counts = df['key_tonic'].value_counts()
            sns.barplot(x=key_counts.index, y=key_counts.values)
            plt.title('Distribution of Musical Keys')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path / 'key_distribution.png')
            plt.close()
        
        # 3. Complexity Features
        complexity_features = [
            'pitch_class_entropy', 'ioi_entropy', 'interval_entropy',
            'chord_entropy', 'token_entropy'
        ]
        
        for feature in complexity_features:
            if feature in df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=feature, kde=True, bins=50)
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
                plt.savefig(save_path / f'{feature}_distribution.png')
                plt.close()
        
        # 4. Correlation Analysis
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            plt.figure(figsize=(16, 12))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(save_path / 'feature_correlations.png')
            plt.close()
        
        logger.info(f"Plots saved to {save_path}")
        
    def save_features(self, output_file: str = 'symbolic_features.json'):
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
    analyzer = SymbolicAnalyzer('data/maestro-v3.0.0')
    
    # Process all files
    analyzer.process_all_files()
    
    # Compute statistics
    analyzer.compute_statistics()
    
    # Create visualizations
    analyzer.plot_distributions()
    
    # Save features
    analyzer.save_features()
    
    # Print comprehensive statistics
    if analyzer.stats:
        print("\n--- Comprehensive Statistics ---")
        for key, value in analyzer.stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        print(f"  {sub_key}: {sub_value:.2f}")
                    else:
                        print(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    else:
        print("No statistics were computed.")

if __name__ == "__main__":
    main()