#!/usr/bin/env python3
"""Quick test to see what FineWeb-2 configurations are actually available"""

from datasets import get_dataset_config_names

print("Fetching FineWeb-2 configuration names...\n")

try:
    configs = get_dataset_config_names("HuggingFaceFW/fineweb-2")

    print(f"Total configurations: {len(configs)}\n")

    # Find English configs
    english_configs = [c for c in configs if 'eng' in c.lower()]
    print(f"English-related configurations ({len(english_configs)}):")
    for config in sorted(english_configs):
        print(f"  - {config}")
    print()

    # Find Thai configs
    thai_configs = [c for c in configs if 'tha' in c.lower() or 'thai' in c.lower()]
    print(f"Thai-related configurations ({len(thai_configs)}):")
    for config in sorted(thai_configs):
        print(f"  - {config}")
    print()

    # Show first 20 configs as examples
    print(f"First 20 configurations (alphabetically):")
    for config in sorted(configs)[:20]:
        print(f"  - {config}")

except Exception as e:
    print(f"Error: {e}")
