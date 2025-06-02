# midi_processing_utils.py
from pathlib import Path
import music21
from music21 import converter, note, chord, tempo, stream
import numpy as np


def extract_key_features_from_score(score: music21.stream.Score) -> dict:
    try:
        key_analysis = score.analyze('key')
        return {
            'key_tonic': key_analysis.tonic.name,
            'key_mode': key_analysis.mode,
            'key_confidence': getattr(key_analysis, 'correlationCoefficient', None)
        }
    except Exception:
        return {'key_tonic': None, 'key_mode': None, 'key_confidence': None}

def extract_pitch_features_from_score(score: music21.stream.Score) -> dict:
    pitches_midi = []
    try:
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                pitches_midi.append(element.pitch.ps)
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    pitches_midi.append(p.ps)
        
        if not pitches_midi:
            return {'num_notes': 0, 'pitch_min': None, 'pitch_max': None, 
                    'pitch_mean': None, 'pitch_median': None, 'pitch_std': None, 'pitch_range': None}

        pitches_array = np.array(pitches_midi)
        return {
            'num_notes': len(pitches_array),
            'pitch_min': float(np.min(pitches_array)) if len(pitches_array) > 0 else None,
            'pitch_max': float(np.max(pitches_array)) if len(pitches_array) > 0 else None,
            'pitch_mean': float(np.mean(pitches_array)) if len(pitches_array) > 0 else None,
            'pitch_median': float(np.median(pitches_array)) if len(pitches_array) > 0 else None,
            'pitch_std': float(np.std(pitches_array)) if len(pitches_array) > 0 else None,
            'pitch_range': float(np.max(pitches_array) - np.min(pitches_array)) if len(pitches_array) > 0 else None
        }
    except Exception:
        return {'num_notes': 0, 'pitch_min': None, 'pitch_max': None, 
                'pitch_mean': None, 'pitch_median': None, 'pitch_std': None, 'pitch_range': None}

def extract_rhythmic_features_from_score(score: music21.stream.Score) -> dict:
    durations_ql = []
    iois_ql = []
    last_onset_offset = None 
    try:
        notes_and_chords_for_ioi = []
        for element in score.flatten().notesAndRests:
            durations_ql.append(element.duration.quarterLength)
            if isinstance(element, (note.Note, chord.Chord)):
                 notes_and_chords_for_ioi.append(element)
        
        # Ensure consistent sorting for IOI calculation, especially with ties in offset
        notes_and_chords_for_ioi.sort(key=lambda x: (x.offset, x.pitch.ps if isinstance(x, note.Note) else (x.pitches[0].ps if isinstance(x, chord.Chord) and x.pitches else 0)))

        for i in range(len(notes_and_chords_for_ioi)):
            current_onset = notes_and_chords_for_ioi[i].offset
            if last_onset_offset is not None:
                ioi = current_onset - last_onset_offset
                if ioi > 0: 
                    iois_ql.append(float(ioi)) # Ensure float for numpy operations
            last_onset_offset = current_onset
            
        return {
            'avg_duration_ql': float(np.mean(durations_ql)) if durations_ql else None,
            'median_duration_ql': float(np.median(durations_ql)) if durations_ql else None,
            'std_duration_ql': float(np.std(durations_ql)) if durations_ql else None,
            'num_rhythmic_elements': len(durations_ql),
            'avg_ioi_ql': float(np.mean(iois_ql)) if iois_ql else None,
            'median_ioi_ql': float(np.median(iois_ql)) if iois_ql else None,
            'std_ioi_ql': float(np.std(iois_ql)) if iois_ql else None,
            'num_iois': len(iois_ql),
            'durations_ql_sample': [float(d) for d in durations_ql[:1000]], # Ensure floats
            'iois_ql_sample': [float(i) for i in iois_ql[:1000]] # Ensure floats
        }
    except Exception as e:
        # It's good to log the error or print it for debugging if needed
        # print(f"Error in extract_rhythmic_features_from_score: {e}")
        return {'avg_duration_ql': None, 'median_duration_ql': None, 'std_duration_ql': None, 'num_rhythmic_elements': 0,
        'avg_ioi_ql': None, 'median_ioi_ql': None, 'std_ioi_ql': None, 'num_iois': 0, 'durations_ql_sample': [], 'iois_ql_sample': []}

def extract_tempo_features_from_score(score: music21.stream.Score) -> dict:
    tempos_bpm = []
    try:
        for mm_obj in score.flatten().getElementsByClass(tempo.MetronomeMark):
            if mm_obj.number: # mm.number is BPM
                tempos_bpm.append(mm_obj.number)
        
        initial_tempo = float(tempos_bpm[0]) if tempos_bpm else None # Taking the first one as initial
        mean_tempo = float(np.mean(tempos_bpm)) if tempos_bpm else None
        num_distinct_tempos = len(set(tempos_bpm))
        
        return {'initial_tempo_bpm': initial_tempo, 'mean_tempo_bpm': mean_tempo, 'num_distinct_tempos': num_distinct_tempos, 'all_tempos_bpm_sample': tempos_bpm[:10]}
    except Exception:
        return {'initial_tempo_bpm': None, 'mean_tempo_bpm': None, 'num_distinct_tempos': 0, 'all_tempos_bpm_sample': []}

def extract_polyphony_features_from_score(score: music21.stream.Score) -> dict:
    try:
        chordified_score = score.chordify()
        polyphony_levels = []

        if chordified_score:
            for element in chordified_score.flatten().notes: # notes includes chords after chordify
                if isinstance(element, chord.Chord):
                    polyphony_levels.append(len(element.pitches))
                elif isinstance(element, note.Note):
                    polyphony_levels.append(1)
        
        # Fallback if chordify results in no notes but original score had notes (e.g. monophonic line)
        if not polyphony_levels and any(score.flatten().getElementsByClass(note.Note)):
            polyphony_levels = [1] * len(list(score.flatten().getElementsByClass(note.Note)))


        if not polyphony_levels:
             return {'avg_polyphony': 0.0, 'max_polyphony': 0, 'median_polyphony': 0.0, 'polyphony_levels_sample': []}

        return {
            'avg_polyphony': float(np.mean(polyphony_levels)) if polyphony_levels else 0.0,
            'max_polyphony': int(np.max(polyphony_levels)) if polyphony_levels else 0,
            'median_polyphony': float(np.median(polyphony_levels)) if polyphony_levels else 0.0,
            'polyphony_levels_sample': polyphony_levels[:1000]
        }
    except Exception:
        return {'avg_polyphony': None, 'max_polyphony': None, 'median_polyphony': None, 'polyphony_levels_sample': []}


def process_single_midi_file(midi_filepath_str: str) -> dict:
    midi_filepath = Path(midi_filepath_str)
    try:
        score = converter.parse(midi_filepath_str, quantizePost=False)
        if score is None:
            return {'midi_filename_processed': midi_filepath.name, 'error': 'music21.converter.parse returned None'}

        # Truncate to one minute (assuming 120 BPM = 480 quarter lengths per minute)
        one_minute_ql = 480.0
        if score.highestTime > one_minute_ql:
            score = score.measures(0, int(one_minute_ql/4))  # Divide by 4 as measures are typically 4 quarter lengths
            truncated = True
        else:
            truncated = False

        features = {
            'midi_filename_processed': midi_filepath.name, 
            'error': None,
            'truncated_to_one_minute': truncated
        }
        features.update(extract_key_features_from_score(score))
        features.update(extract_pitch_features_from_score(score))
        features.update(extract_rhythmic_features_from_score(score))
        features.update(extract_tempo_features_from_score(score))
        features.update(extract_polyphony_features_from_score(score))
        
        total_ql = 0.0
        if hasattr(score, 'duration') and hasattr(score.duration, 'quarterLength'):
            total_ql = score.duration.quarterLength
        elif score.highestTime > 0 : # Fallback if .duration is not populated as expected
             total_ql = score.highestTime
        features['total_quarter_length'] = float(total_ql)

        return features
    except Exception as e:
        import traceback
        error_str = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return {'midi_filename_processed': midi_filepath.name, 'error': str(e)}