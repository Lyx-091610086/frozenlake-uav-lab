"""Build browser data without importing or executing the submitted training code."""
from pathlib import Path
import ast, json, math, statistics, hashlib
ROOT=Path(__file__).resolve().parents[1]
def literal_assignment(path,name):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'))
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
 raise ValueError(name)
def main():
 maps=literal_assignment(ROOT/'original/benchmark/environments/maps_icegrid.py','MAPS')
 original=ROOT/'original/benchmark/results/experiment_summary.json'
 records=json.loads(original.read_text())
 groups={}
 for r in records:
  assert 0<=r['success_rate']<=1 and math.isfinite(r['avg_steps'])
  key=(r['map'],r['slippery'],r['algorithm'])
  groups.setdefault(key,[]).append(r)
 summary=[]
 for (map_name,slippery,algorithm),rows in groups.items():
  rates=[r['success_rate'] for r in rows]
  summary.append(dict(map=map_name,slippery=slippery,algorithm=algorithm,n=len(rows),success=statistics.mean(rates),std=statistics.stdev(rates) if len(rows)>1 else 0,steps=statistics.mean(r['avg_steps'] for r in rows),seconds=statistics.mean(r['training_time'] for r in rows),rates=rates))
 payload={'maps':maps,'benchmarks':summary,'meta':{'records':len(records),'algorithms':sorted({r['algorithm'] for r in records}),'definedMaps':len(maps),'evaluatedMaps':len({r['map'] for r in records}),'seeds':sorted({int(r['experiment_id'].rsplit('_',1)[1]) for r in records}),'summarySha256':hashlib.sha256(original.read_bytes()).hexdigest(),'metric':'For learning algorithms, max of checkpoint-evaluation and final-evaluation success; planning uses final evaluation. Original settings differ across algorithms.'}}
 (ROOT/'docs/data/archive.js').write_text('window.FROZEN_ARCHIVE = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n')
 (ROOT/'docs/data/experiment_summary.json').write_bytes(original.read_bytes())
 print(json.dumps(payload['meta'],indent=2))
if __name__=='__main__':main()
