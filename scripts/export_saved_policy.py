"""Read tensor-only PyTorch ZIP checkpoints with a restricted metadata decoder.

This does not import the submitted scripts or use unrestricted pickle loading.
Unknown pickle globals, storage types or tensor layouts are rejected.
Only float32 tensors in the provided fully connected network are supported.
"""
from pathlib import Path
import io,zipfile,pickle,collections,json,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class Tensor:
 def __init__(self,storage,offset,shape,stride,*args):self.storage,self.offset,self.shape,self.stride=storage,offset,shape,stride
class Decoder(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return Tensor
  if (module,name)==('torch','FloatStorage'):return 'float32'
  raise ValueError('Unsupported checkpoint global: '+module+'.'+name)
 def persistent_load(self,pid):
  if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or pid[1]!='float32':raise ValueError('Unsupported storage')
  return {'key':pid[2],'size':pid[4]}
def load_tensors(path):
 with zipfile.ZipFile(path) as z:
  meta=next(n for n in z.namelist() if n.endswith('/data.pkl'));prefix=meta[:-8]
  if prefix+'byteorder' in z.namelist() and z.read(prefix+'byteorder')!=b'little':raise ValueError('Byteorder')
  params=Decoder(io.BytesIO(z.read(meta))).load();result={}
  for name,t in params.items():
   if not isinstance(t,Tensor):raise ValueError('Not a tensor')
   flat=np.frombuffer(z.read(prefix+'data/'+t.storage['key']),dtype='<f4')
   expected=tuple(int(np.prod(t.shape[i+1:])) for i in range(len(t.shape)))
   if tuple(t.stride)!=expected:raise ValueError('Non-contiguous tensor')
   result[name]=flat[t.offset:t.offset+int(np.prod(t.shape))].reshape(t.shape).copy()
  return result
def infer(params,states):
 x=np.array(states,dtype=np.float32)
 for i in [0,2,4,6]:
  x=x@params[f'model.{i}.weight'].T+params[f'model.{i}.bias']
  if i!=6:x=np.maximum(x,0)
 return x
SMALL=[[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,1,0,1,1],[1,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,1,1,1]]
def main():
 base=ROOT/'original/uav/模型权重/ddqn_s.pth';params=load_tensors(base)
 assert params['model.0.weight'].shape==(512,36)
 states=[]
 for row in range(1,5):
  for col in range(1,5):
   board=np.array(SMALL,dtype=np.float32);board[row,col]=2;board[4,4]=3;states.append(board.flatten())
 q=infer(params,states)
 # Source action order up/down/left/right -> Gym order left/down/right/up.
 converted=q[:,[2,1,3,0]]
 desc=[]
 for r in range(1,5):
  desc.append(''.join('S' if (r,c)==(1,1) else 'G' if (r,c)==(4,4) else 'W' if SMALL[r][c] else 'F' for c in range(1,5)))
 payload={'name':'uav_checkpoint_small','map':desc,'qValues':converted.round(6).tolist(),'policy':converted.argmax(axis=1).tolist(),'source':'ddqn_s.pth','sourceSha256':hashlib.sha256(base.read_bytes()).hexdigest(),'collision':'blocked-wall','actionOrder':['left','down','right','up'],'note':'Finite-state inference export of the supplied checkpoint. Its source training script uses a DQN target despite the ddqn filename. This wall-navigation environment differs from terminal-hole FrozenLake.'}
 (ROOT/'docs/data/checkpoint.js').write_text('window.SAVED_POLICY = '+json.dumps(payload,separators=(',',':'))+';\n')
 print('Exported',len(q),'states; checkpoint shape',params['model.0.weight'].shape)
 print('Policy:',payload['policy'],'map:',desc)
if __name__=='__main__':main()
