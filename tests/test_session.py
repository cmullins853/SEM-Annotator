"""Session store checks (resilience spec §7). Run: python -m tests.test_session"""
import json
import tempfile
from pathlib import Path

from fibersight import session
from fibersight.state import AppState


def test_round_trip_and_validation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / 'session.json'
        state = AppState(image_dir=str(tmp), selected_image='a.tif', scale_mode='locked',
                         scale_um_per_px=0.0123456789, roi_crop_bottom_pct=12.0)

        assert session.save_session(session.build_payload(state), path)
        loaded = session.load_session(path)
        assert loaded is not None
        assert loaded['scale_um_per_px'] == 0.0123456789          # lossless (REQ-48/53)
        assert not list(tmp.glob('*.tmp'))                        # atomic write cleaned up

        for bad in ('[]', '{"version": 999}', '{trunc'):          # REQ-52/57, R-G9
            path.write_text(bad)
            assert session.load_session(path) is None

        state.image_dir = str(tmp / 'gone')                       # folder deleted -> no offer
        session.save_session(session.build_payload(state), path)
        assert session.load_session(path) is None

        session.clear_session(path)
        assert not path.exists()
        assert session.load_session(path) is None                 # missing file: no crash


def test_image_scale_store_keeps_other_entries():
    with tempfile.TemporaryDirectory() as tmp:
        session.save_image_scale(tmp, 'a.tif', 0.5, 10.0, 20.0)
        session.save_image_scale(tmp, 'b.tif', 0.25, 5.0, 20.0)
        assert session.load_image_scale(tmp, 'a.tif')['scale_um_per_px'] == 0.5
        assert session.load_image_scale(tmp, 'b.tif')['scale_um_per_px'] == 0.25
        assert session.load_image_scale(tmp, 'missing.tif') is None
        data = json.loads((Path(tmp) / '.fibersight' / 'session.json').read_text())
        assert set(data['images']) == {'a.tif', 'b.tif'}


def test_restore_applies_locked_scale():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ('a.png', 'b.png'):                  # .png: no background preview work
            (Path(tmp) / name).write_bytes(b'')
        saved = {'version': session.SESSION_VERSION, 'image_dir': tmp, 'selected_image': 'b.png',
                 'scale_mode': 'locked', 'scale_um_per_px': 0.3, 'advanced_mode': True}
        state = AppState()
        state.restore(saved)
        assert state.image_files == ['a.png', 'b.png']
        assert state.selected_image == 'b.png'
        assert state.scale_um_per_px == 0.3 and state.advanced_mode


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
