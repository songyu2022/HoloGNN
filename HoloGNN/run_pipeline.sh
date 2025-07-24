#!/bin/bash

docking_dir="$1"
csv_file='unidock_result.csv'

if [ -z "$docking_dir" ]; then
  echo "❌ Error: docking_dir parameter missing."
  echo "Usage: bash run_pipeline.sh <docking_dir>"
  exit 1
fi

echo "🔁 Running step1 "
python preprocessing_docking.py --docking_dir "$docking_dir" --csv_file "$csv_file"

echo "🔁 Running step2 "
python dataset_GIGN_docking.py --docking_dir "$docking_dir" --csv_file "$csv_file"

echo "🔁 Running step3 "
python predict_batch.py --docking_dir "$docking_dir" --csv_file "$csv_file"

echo "✅ All scripts completed."
