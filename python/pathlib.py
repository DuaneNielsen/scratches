from pathlib import Path
import pathlib

q = Path(r'c:\temp')

# recursively list files
filelist = list(q.glob('**/*.py'))

# create parent directory if it doesn't exist
q.parent.mkdir(parents=True, exist_ok=True)

# get the name of the file only
q.name

# check if file exists
q.exists()

# open file
with q.open() as f:
    f.readline()

# strip suffix from file
q = q.with_suffix('')

# return the file string
str(q)

# get a list of files in a directory
directory = 'c:\\'
files = list(Path(directory).glob('*.png'))

# create a uniq file
def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path

path = unique_path(pathlib.Path.cwd(), 'test{:03d}.txt')

# get the latest file
from datetime import datetime
time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
print(datetime.fromtimestamp(time), file_path)

# display a directory tree
def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')

tree(pathlib.Path.cwd())