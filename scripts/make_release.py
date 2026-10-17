"""Package the complete clean project, excluding generated output and host metadata."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
output=root/'docs/assets/frozenlake-project.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if p.is_file() and p!=output and not any(x in {'.git','.openai','__pycache__','dist','.venv','runs'}for x in rel.parts):z.write(p,Path('frozenlake-lab')/rel)
assert output.stat().st_size<25*1024*1024,'Static download exceeds per-file host limit'
print('Packaged complete source, models, data, reports and videos:',output.stat().st_size,'bytes')
