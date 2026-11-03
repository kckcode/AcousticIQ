# MIMII Dataset Documentation

## 1. Official Dataset Record

- **Record Title:** Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection (MIMII Dataset)
- **Official URL:** https://zenodo.org/records/3384388
- **Authors:** Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, Yohei Kawaguchi (Hitachi, Ltd.)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## 2. Machine Categories & Failure Modes

The MIMII dataset comprises audio recordings from four industrial machine categories:

1. **Valves:**
   - Operating dynamics: Fast solenoid opening/closing strokes with pneumatic flow.
   - Failure modes: Air leakage, mechanical friction, contamination, delayed seating chatter.
2. **Pumps:**
   - Operating dynamics: Centrifugal motor operation with 50 Hz power harmonics.
   - Failure modes: Impeller cavitation, clogging, bearing race spall, rotor unbalance.
3. **Fans:**
   - Operating dynamics: Multi-blade rotation with Blade Pass Frequency (BPF = blades × RPM/60) and turbulence.
   - Failure modes: Blade deformation, airflow blockage, unbalanced shaft flutter.
4. **Slide Rails (Sliders):**
   - Operating dynamics: Linear reciprocating motion with acceleration, constant velocity, and deceleration.
   - Failure modes: Rail surface scoring, insufficient lubrication, mechanical sticking.

## 3. Signal-to-Noise Ratios (SNR) & Acoustic Conditions

Real factory background noise recordings were blended with machine sounds at three distinct SNR levels:
- **6 dB:** Relatively clean industrial environment.
- **0 dB:** Moderate factory background noise where noise energy equals machine signal energy.
- **-6 dB:** Severe ambient noise environment where machine sounds are partially masked.

## 4. Archive Organization

The official release is partitioned into 12 ZIP archives:
- `6_dB_valve.zip` (6.9 GB), `0_dB_valve.zip` (7.5 GB), `-6_dB_valve.zip` (8.0 GB)
- `6_dB_pump.zip` (7.7 GB), `0_dB_pump.zip` (7.9 GB), `-6_dB_pump.zip` (8.2 GB)
- `6_dB_fan.zip` (10.2 GB), `0_dB_fan.zip` (10.4 GB), `-6_dB_fan.zip` (10.9 GB)
- `6_dB_slider.zip` (7.1 GB), `0_dB_slider.zip` (7.5 GB), `-6_dB_slider.zip` (8.0 GB)

## 5. Local Downloader & Calibrated Synthetic Fixtures

To avoid unconditional multi-gigabyte downloads during rapid testing, AcousticIQ provides:
1. `scripts/download_mimii.py`: Targeted resumable downloader supporting selective machine category acquisition.
2. `ml/data/synthetic_fixture.py`: Physics-based mathematical machine sound generator reproducing rotational, pneumatic, and cavitation dynamics for CI smoke tests. Synthetic fixtures are explicitly tagged with `is_synthetic: true` to preserve scientific integrity.
