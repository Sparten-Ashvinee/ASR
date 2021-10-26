from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

def audiomentations_lib():

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    return augment


def aug_effects(SAMPLE_RATE):

    # Define effects
    effects = [
      ["lowpass", "-1", "300"], # apply single-pole lowpass filter
      ["speed", "0.8"],  # reduce the speed
                         # This only changes sample rate, so it is necessary to
                         # add `rate` effect with original sample rate after this.
      ["rate", f"{SAMPLE_RATE}"],
      ["reverb", "-w"],  # Reverbration gives some dramatic feeling
    ]

    return effects


