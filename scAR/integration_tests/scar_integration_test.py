from xml.dom import ValidationErr
from scAR import model
import pandas as pd
import sys
import os

raw_count = pd.read_pickle('../test/raw_counts.pickle')
empty_profile = pd.read_pickle('../test/ambient_profile.pickle')
expected_output = pd.read_pickle('../test/output_assignment.pickle')

scarObj = model(raw_count=raw_count.values,
                empty_profile=empty_profile,
                scRNAseq_tech='CROPseq')

scarObj.train(epochs=40,
              batch_size=64,)

scarObj.inference()

if scarObj.feature_assignment.equals(expected_output):
    sys.stdout.write(f"Successful integration test.{os.linesep}")
else:
    raise ValidationErr(f"Error in integration test.{os.linesep}")