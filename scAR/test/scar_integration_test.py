from scAR import model
import pandas as pd
import sys
import os

raw_count = pd.read_pickle('raw_counts.pickle')
empty_profile = pd.read_pickle('ambient_profile.pickle')
expected_output = pd.read_pickle('output_assignment.pickle')

scarObj = model(raw_count=raw_count.values,
                empty_profile=empty_profile,
                scRNAseq_tech='CROPseq')

scarObj.train(epochs=40,
              batch_size=64,)

scarObj.inference()

status = scarObj.feature_assignment.equals(expected_output)

if status:
    sys.stdout.write(f"Successful integration test.{os.linesep}")
    sys.exit(0)
else:
    sys.stdout.write(f"Error in integration test.{os.linesep}")
    sys.exit(1)