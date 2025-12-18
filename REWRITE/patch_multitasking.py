import pathlib, re
candidates = [
    pathlib.Path('/usr/local/lib/python3.8/dist-packages'),
    pathlib.Path('/usr/local/lib/python3.9/dist-packages'),
    pathlib.Path('/usr/local/lib/python3.8/site-packages'),
]
for base in candidates:
    f = base / 'multitasking' / '__init__.py'
    if f.exists():
        txt = f.read_text()
        if 'from typing import Type' not in txt:
            txt = 'from typing import Type\n' + txt
        txt = re.sub(r'type\\[([^\\]]+)\\]', r'Type[\\1]', txt)
        f.write_text(txt)
        print('patched', f)
        break
else:
    print('multitasking __init__.py not found, skipping patch')