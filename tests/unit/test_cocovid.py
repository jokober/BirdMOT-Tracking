from src.sahi_tracking.formats.cocovid import filter_sequences
from tests.fixtures.fixtures import primary_sequence_dict_fixture


def test_cocovid_filter(tmp_path, primary_sequence_dict_fixture):
    assert 'C0054_783015_783046' in [sequence['name'] for sequence in primary_sequence_dict_fixture['videos']]

    filtered_cocovid = filter_sequences(primary_sequence_dict_fixture, ['C0054_783015_783046'])
    assert 'C0054_783015_783046' in [sequence['name'] for sequence in filtered_cocovid['videos']]

    filtered_cocovid = filter_sequences(primary_sequence_dict_fixture, [])
    assert 'C0054_783015_783046' not in [sequence['name'] for sequence in filtered_cocovid['videos']]