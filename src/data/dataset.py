"""
Multi-subject ECoG/EEG dataset for motor imagery classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import mne
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


class MultiSubjectECoGDataset(Dataset):
    """
    PyTorch Dataset for loading multi-subject ECoG/EEG data.

    Loads .fif files, applies preprocessing, and returns (eeg_data, label) pairs.
    Text captions are generated dynamically during training by the model/trainer.

    Args:
        file_list: List of paths to .fif files
        freq_band: Frequency band for filtering (low, high) or None
        remove_bad: Whether to remove bad channels
        scale: Whether to apply standard scaling per channel
        transform: Optional transform to apply to samples
        time_range: Time range for epoching (tmin, tmax) in seconds
    """

    def __init__(
        self,
        file_list: List[str],
        freq_band: Optional[Tuple[float, float]] = (1, 100),
        remove_bad: bool = True,
        scale: bool = True,
        transform=None,
        time_range: Optional[Tuple[float, float]] = None
    ):
        self.transform = transform
        self.data = []
        self.labels = []
        self.scale = scale
        self.remove_bad = remove_bad
        self.freq_band = freq_band
        self.time_range = time_range

        # Load all subject data
        for file in file_list:
            print(f'Loading and processing {file}')
            self._load_subject_data(file)

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        print(f"\nDataset Summary:")
        print(f"  Loaded {len(self.data)} trials from {len(file_list)} subjects")
        print(f"  Data shape: {self.data.shape}")

        # Print class distribution
        self._print_class_distribution()

    def _load_subject_data(self, file_path: str):
        """Load and preprocess data from a single subject file"""
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        if self.remove_bad:
            raw.pick(['ecog', 'eeg'], exclude='bads')

        # Apply bandpass filter
        if self.freq_band:
            raw.filter(
                self.freq_band[0],
                self.freq_band[1],
                fir_design='firwin',
                verbose=False
            )

        # Get events and labels
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        print(f"  Found {len(events)} events")

        # Epoching
        if self.time_range:
            tmin, tmax = self.time_range
        else:
            tmin, tmax = 0, 4

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        # Get data and labels
        X = epochs.get_data()
        y = epochs.events[:, -1]

        # Scaling per channel
        if self.scale:
            for ch in range(X.shape[1]):
                scaler = StandardScaler()
                X[:, ch, :] = scaler.fit_transform(X[:, ch, :])

        # Save data and labels
        for trial, label in zip(X, y):
            self.data.append(trial)
            # Convert to 0-indexed (0=left, 1=right, 2=foot, 3=rest)
            self.labels.append(int(label) - 1)

    def _print_class_distribution(self):
        """Print class distribution"""
        unique, counts = np.unique(self.labels, return_counts=True)
        class_names = ["left hand", "right hand", "foot", "rest"]

        print(f"\nClass distribution:")
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                print(f"  Class {cls} ({class_names[cls]}): {count} samples")
            else:
                print(f"  Class {cls}: {count} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            index: Sample index

        Returns:
            sample: EEG data tensor (1, num_electrodes, time_points)
            label: Class label (0-3)
        """
        sample = self.data[index]
        label = self.labels[index]

        # Convert to tensor and add channel dimension
        sample = torch.from_numpy(sample).float().unsqueeze(0)
        label = torch.tensor(label).long()

        # Apply transform if any
        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def get_num_channels(self) -> int:
        """Get number of EEG channels"""
        return self.data.shape[1]

    def get_time_points(self) -> int:
        """Get number of time points"""
        return self.data.shape[2]

    def get_class_counts(self) -> dict:
        """Get count of samples per class"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
