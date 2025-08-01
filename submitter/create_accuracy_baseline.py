#!/usr/bin/env python3
import json, sys

def main():
    if len(sys.argv) != 4:
        print("Usage: create_accuracy_baseline.py <accuracy_json> <compliance_json> <output_json>", file=sys.stderr)
        return 1
    acc_path, comp_path, out_path = sys.argv[1:4]
    with open(acc_path) as f:
        acc_data = json.load(f)
    with open(comp_path) as f:
        comp_data = json.load(f)
    comp_indices = {item.get('qsl_idx') for item in comp_data.get('results', [])}
    baseline_results = [item for item in acc_data.get('results', []) if item.get('qsl_idx') in comp_indices]
    with open(out_path, 'w') as f:
        json.dump({'results': baseline_results}, f)
    return 0

if __name__ == '__main__':
    sys.exit(main())
