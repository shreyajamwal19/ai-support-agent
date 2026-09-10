import json


def test_taxonomy_json_valid_and_well_formed():
    tax = json.load(open("configs/intent_taxonomy.json"))
    assert "intents" in tax
    ids = [i["id"] for i in tax["intents"]]
    assert len(ids) == len(set(ids)), "duplicate intent ids"
    assert "OTHER_UNCLEAR" in ids
    for intent in tax["intents"]:
        for field in ["id", "name", "description", "inclusion_criteria",
                       "exclusion_criteria", "representative_examples", "common_confusions"]:
            assert field in intent, f"{intent.get('id')} missing field {field}"
        assert len(intent["representative_examples"]) >= 1
