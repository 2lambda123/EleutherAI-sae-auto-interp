import os
import pytest
import json

filename = ".transformer.h.0_feature0.json"

@pytest.fixture
def record():
    from sae_auto_interp.features import FeatureRecord, FeatureID
    from sae_auto_interp.features.example import Example

    current_dir = os.path.dirname(__file__)  # Gets the directory of the current script
    file_path = os.path.join(current_dir, filename)
    with open(file_path, 'r') as file:
        content = json.load(file)['examples']

    record = FeatureRecord(
        FeatureID(
            ".transformer.h.0",
            0
        )
    )

    record.examples = [
        Example(
            tokens=i['tokens'],
            activations=i['activations'],
        ) 
        for i in content
    ]

    assert len(record.examples) == 100
    assert str(record.feature) == ".transformer.h.0_feature0"

    return record

@pytest.mark.usefixtures("record")
def test_record():

    pass

def test_sample_top_and_quantiles(record):
    from sae_auto_interp.features.sampling import sample_top_and_quantiles
    print(len(record.examples   ))
    sample_top_and_quantiles(record)

    assert len(record.train) == 10
    assert len(record.test) == 4
    assert len(record.test[0]) == 5

