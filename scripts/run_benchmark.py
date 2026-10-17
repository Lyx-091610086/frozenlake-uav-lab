"""Small explicit entry point for the archived benchmark; writes new results separately."""
from pathlib import Path
import argparse,sys,copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'original/benchmark'))
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--algorithm',default='q_learning')
 parser.add_argument('--map',default='4x4_default')
 parser.add_argument('--episodes',type=int,default=1000)
 parser.add_argument('--seed',type=int,default=42)
 parser.add_argument('--deterministic',action='store_true')
 parser.add_argument('--results-dir',type=Path,default=ROOT/'runs')
 args=parser.parse_args()
 from experiments.experiment_runner import ExperimentRunner
 from experiments.configs import ALGORITHM_CONFIGS
 runner=ExperimentRunner(str(args.results_dir),save_detailed_results=True)
 if args.algorithm not in runner.algorithms:parser.error('Algorithm unavailable; deep methods also require PyTorch. Available: '+', '.join(runner.algorithms))
 config=copy.deepcopy(ALGORITHM_CONFIGS.get(args.algorithm,{}))
 config.update(max_episodes=args.episodes,eval_every=max(100,args.episodes//5),eval_episodes=100)
 result=runner.run_single_experiment(args.algorithm,args.map,not args.deterministic,config,args.seed)
 print('FINAL evaluation (separate from checkpoint maximum):',result['final_evaluation'])
 print('Original runner retains its historical model and seed handling. See AUDIT.md before comparing results.')
if __name__=='__main__':main()
