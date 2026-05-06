"""
Caption generation for EEG trials.
Supports both fixed templates (for 4-class) and varied templates (for batch contrastive).
"""

import numpy as np
from typing import List, Dict


class ClassTemplates:
    """
    Fixed class templates for 4-class EEG-CLIP.
    Returns one template per class (no variation).
    """

    TEMPLATE_STYLES = {
        'simple': [
            "left hand",
            "right hand",
            "foot",
            "rest"
        ],
        'action_focused': [
            "left hand movement",
            "right hand movement",
            "foot movement",
            "resting state"
        ],
        'descriptive': [
            "person performing left hand grasping motion",
            "person performing right hand grasping motion",
            "person performing foot dorsiflexion movement",
            "person in relaxed resting state"
        ]
    }

    def __init__(self, template_style: str = 'action_focused'):
        """
        Args:
            template_style: Template style to use
                Options: 'simple', 'action_focused', 'descriptive'
        """
        if template_style not in self.TEMPLATE_STYLES:
            print(f"Warning: Unknown template style '{template_style}', using 'action_focused'")
            template_style = 'action_focused'

        self.template_style = template_style
        self.templates = self.TEMPLATE_STYLES[template_style]

        print(f"\n{'='*80}")
        print(f"CLASS TEMPLATES - Style: '{template_style}'")
        print(f"{'='*80}")
        for i, template in enumerate(self.templates):
            print(f"  Class {i}: '{template}'")
        print(f"{'='*80}\n")

    def get_class_caption(self, class_idx: int) -> str:
        """Get caption for a specific class index (0-3)"""
        return self.templates[class_idx]

    def get_all_captions(self) -> List[str]:
        """Get all 4 class captions"""
        return self.templates


class CaptionGenerator:
    """
    Generates varied text captions for EEG trials.
    Each sample can get a unique caption based on its class.
    Used for batch-wise contrastive learning.
    """

    TEMPLATE_STYLES = {
        'simple': {
            0: ["left hand"],
            1: ["right hand"],
            2: ["foot"],
            3: ["rest"]
        },
        'varied': {
            0: [
                "left hand movement",
                "moving left hand",
                "performing left hand movement",
                "left hand grasping motion",
                "left hand action",
                "executing left hand movement"
            ],
            1: [
                "right hand movement",
                "moving right hand",
                "performing right hand movement",
                "right hand grasping motion",
                "right hand action",
                "executing right hand movement"
            ],
            2: [
                "foot movement",
                "moving foot",
                "performing foot movement",
                "foot dorsiflexion",
                "foot action",
                "executing foot movement"
            ],
            3: [
                "resting state",
                "no movement",
                "rest",
                "relaxed state",
                "no motor task",
                "baseline activity"
            ]
        },
        'descriptive': {
            0: [
                "person performing left hand grasping motion",
                "brain activity during left hand movement",
                "neural patterns of left hand movement",
                "EEG signals from left hand task",
                "cortical activity for left hand motion"
            ],
            1: [
                "person performing right hand grasping motion",
                "brain activity during right hand movement",
                "neural patterns of right hand movement",
                "EEG signals from right hand task",
                "cortical activity for right hand motion"
            ],
            2: [
                "person performing foot dorsiflexion movement",
                "brain activity during foot movement",
                "neural patterns of foot movement",
                "EEG signals from foot task",
                "cortical activity for foot motion"
            ],
            3: [
                "person in relaxed resting state",
                "brain activity during rest",
                "baseline neural patterns",
                "EEG signals during rest",
                "cortical activity at rest"
            ]
        }
    }

    def __init__(self, template_style: str = 'varied'):
        """
        Args:
            template_style: Template style to use
                Options: 'simple', 'varied', 'descriptive'
        """
        if template_style not in self.TEMPLATE_STYLES:
            print(f"Warning: Unknown template style '{template_style}', using 'varied'")
            template_style = 'varied'

        self.template_style = template_style
        self.templates = self.TEMPLATE_STYLES[template_style]

        print(f"\n{'='*80}")
        print(f"CAPTION GENERATOR - Style: '{template_style}'")
        print(f"{'='*80}")
        class_names = ["left hand", "right hand", "foot", "rest"]
        for class_idx, templates in self.templates.items():
            print(f"  Class {class_idx} ({class_names[class_idx]}): {len(templates)} templates")
            for template in templates:
                print(f"    - '{template}'")
        print(f"{'='*80}\n")

    def generate_caption(self, class_idx: int, variant_idx: int = 0) -> str:
        """
        Generate caption for a specific class and variant.

        Args:
            class_idx: Class index (0-3)
            variant_idx: Variant index (cycles through available variants)

        Returns:
            caption: Text caption
        """
        templates = self.templates[class_idx]
        return templates[variant_idx % len(templates)]

    def generate_batch_captions(self, labels: np.ndarray) -> List[str]:
        """
        Generate captions for a batch of labels.
        Uses different variants to create unique captions.

        Args:
            labels: Array of class labels

        Returns:
            captions: List of text captions
        """
        captions = []
        for i, label in enumerate(labels):
            # Use random variant for diversity
            variant_idx = np.random.randint(0, len(self.templates[int(label)]))
            captions.append(self.generate_caption(int(label), variant_idx))
        return captions

    def get_first_variants(self) -> List[str]:
        """
        Get first variant for each class.
        Useful for evaluation with fixed templates.

        Returns:
            captions: List of 4 captions (one per class)
        """
        return [self.generate_caption(i, 0) for i in range(4)]
