import os

import dotenv

DATA_DIR = dotenv.get('DATA_DIR')
ACTIVITIES = [
    'normal_walk_1_0-6',
    'normal_walk_1_1-2',
    'normal_walk_1_1-8',
    'normal_walk_1_2-0',
    'normal_walk_1_2-5',
]
SUBJECTS = [f'AB{i:02}' for i in range(1, 14)]

def _get_data_files() -> dict:
    """
    Helper function to get the list of emg and angle files.
    Returns:
        dict: A dictionary with keys 'emg_files' and 'angle_files' containing lists of file paths.
    """
    files = []

    for subject in SUBJECTS:
        for activity in ACTIVITIES:
            emg_file = os.path.join(DATA_DIR, subject, f'{subject}_{activity}_emg.csv')
            angle_file = os.path.join(DATA_DIR, subject, f'{subject}_{activity}_angle.csv')

            if os.path.exists(emg_file) and os.path.exists(angle_file):
                files.append({
                    'subject': subject,
                    'activity': activity,
                    'emg_file': emg_file,
                    'angle_file': angle_file
                })
            else:
                print(f"Warning: Missing files for {subject} - {activity}")
    return files

DATA_FILES = _get_data_files()