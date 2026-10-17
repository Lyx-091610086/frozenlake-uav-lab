"""Copy the buildless docs site to the hosting output directory."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
source=root/'docs'
assert (source/'index.html').is_file()
output=root/'dist'
if output.exists():shutil.rmtree(output)
shutil.copytree(source,output)
print('Prepared dist from docs')
