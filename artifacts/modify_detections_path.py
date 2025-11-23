#!/usr/bin/env python3
"""
Script to modify image paths in detections.json file.
Replaces absolute paths with relative paths based on season.
"""

import json
import os


def modify_detections_paths(json_path):
    """
    Modify image paths in detections.json file.
    
    Replacements:
    - /Volumes/KAUSAR/rover_dataset/2024-01-13/ -> data/winter/ (if season is winter)
    - /Volumes/KAUSAR/rover_dataset/2024-04-11/ -> data/autumn/ (if season is autumn)
    
    Args:
        json_path: Path to the detections.json file
    """
    # Load the JSON file
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count modifications
    winter_count = 0
    autumn_count = 0
    
    # Iterate through winter and autumn entries
    # Process winter entries
    if 'winter' in data:
        for entry in data['winter']:
            if 'image_path' in entry:
                original_path = entry['image_path']
                if '/Volumes/KAUSAR/rover_dataset/2024-01-13/' in original_path:
                    entry['image_path'] = original_path.replace(
                        '/Volumes/KAUSAR/rover_dataset/2024-01-13/',
                        'data/winter/'
                    )
                    winter_count += 1
    
    # Process autumn entries
    if 'autumn' in data:
        for entry in data['autumn']:
            if 'image_path' in entry:
                original_path = entry['image_path']
                if '/Volumes/KAUSAR/rover_dataset/2024-04-11/' in original_path:
                    entry['image_path'] = original_path.replace(
                        '/Volumes/KAUSAR/rover_dataset/2024-04-11/',
                        'data/autumn/'
                    )
                    autumn_count += 1
    
    # Create backup of original file
    backup_path = json_path + '.backup'
    print(f"Creating backup at {backup_path}...")
    with open(backup_path, 'w') as f:
        json.dump(data, f)
    
    # Save modified data back to original file
    print(f"Saving modified data to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nModification complete!")
    print(f"  - Winter paths modified: {winter_count}")
    print(f"  - Autumn paths modified: {autumn_count}")
    print(f"  - Total modifications: {winter_count + autumn_count}")
    print(f"  - Backup saved to: {backup_path}")


if __name__ == '__main__':
    # Path to detections.json in artifacts folder
    json_path = os.path.join(
        os.path.dirname(__file__),
        'artifacts',
        'detections.json'
    )
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        exit(1)
    
    modify_detections_paths(json_path)
